import random
from functools import partial
from typing import Any, Optional, Union, Dict

from datasets import Dataset
from pytorch_lightning.utilities import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from ..base import TaskDataModule
from ...utils.logger import logger


class EventExtractionDataModule(TaskDataModule):
    def __init__(
        self,
        *args,
        task_name: str = "gplinker",
        is_chinese: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.task_name = task_name
        self.is_chinese = is_chinese

    def get_process_fct(self, text_column_name, label_column_name, mode):
        max_length = self.train_max_length
        if mode in ["val", "test"]:
            max_length = self.validation_max_length if mode == "val" else self.test_max_length

        convert_to_features = partial(
            EventExtractionDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            max_length=max_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            mode=mode,
            is_chinese=self.is_chinese,
        )
        return convert_to_features

    def process_data(self, dataset: Union[Dataset, Dict], stage: Optional[str] = None) -> Union[Dataset, Dict]:
        label_column_name, text_column_name = self._setup_input_fields(dataset, stage)
        self._prepare_labels(dataset, label_column_name)

        convert_to_features_train = self.get_process_fct(text_column_name, label_column_name, "train")
        convert_to_features_val = self.get_process_fct(text_column_name, label_column_name, "val")

        train_dataset = self.process_train(dataset["train"], predicate2id=self.predicate_to_id)
        train_dataset = train_dataset.map(
            convert_to_features_train,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train datasets",
            new_fingerprint=f"train-{self.train_max_length}-{self.task_name}",
            num_proc=self.num_workers,
        )

        def process_dev(example):
            events_label = []
            event_list = example.get("event_list", None)
            if event_list is not None:
                for e in event_list:
                    etype = e["event_type"]
                    role = "触发词"
                    argument = e["trigger"]
                    event = [(etype + "+" + role, argument)]

                    for a in e["arguments"]:
                        role = a["role"]
                        argument = a["argument"]
                        event.append((etype + "+" + role, argument))
                    events_label.append(event)
            return {"text": example["text"], "target": events_label}

        val_dataset = dataset["validation"].map(process_dev)
        val_dataset = val_dataset.map(
            convert_to_features_val,
            batched=True,
            remove_columns=[label_column_name],
            desc="Running tokenizer on validation datasets",
            new_fingerprint=f"validation-{self.validation_max_length}-{self.task_name}",
            num_proc=self.num_workers,
        )

        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Length of training set: {len(train_dataset)}")
            logger.info(f"Sample {index} of the training set:")
            for k, v in train_dataset[index].items():
                logger.info(f"{k} = {v}")

        for index in random.sample(range(len(val_dataset)), 1):
            logger.info(f"Length of validation set: {len(val_dataset)}")
            logger.info(f"Sample {index} of the validation set:")
            for k, v in val_dataset[index].items():
                logger.info(f"{k} = {v}")

        all_dataset = {"train": train_dataset, "validation": val_dataset}

        if "test" in dataset:
            test_dataset = dataset["test"].map(process_dev)
            convert_to_features_test = self.get_process_fct(text_column_name, label_column_name, "test")
            test_dataset = test_dataset.map(
                convert_to_features_test,
                batched=True,
                remove_columns=[label_column_name],
                desc="Running tokenizer on test datasets",
                new_fingerprint=f"test-{self.test_max_length}-{self.task_name}",
                num_proc=self.num_workers,
            )

            all_dataset["test"] = test_dataset

        return all_dataset

    def _setup_input_fields(self, dataset, stage):
        split = "train" if stage == "fit" else "validation"
        column_names = dataset[split].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        label_column_name = "event_list" if "event_list" in column_names else column_names[1]
        return label_column_name, text_column_name

    def _prepare_labels(self, dataset, label_column_name):
        if self.labels is None:
            # Create unique label set from train datasets.
            labels = []
            for line in dataset["train"]:
                roles = ["触发词"] + [o["role"] for o in line["role_list"]]
                labels.extend([line["event_type"] + "+" + role for role in roles])
            self.labels = list(set(labels))

        self.labels = sorted(self.labels)
        self.predicate_to_id = {l: i for i, l in enumerate(self.labels)}

    @property
    def schemas(self):
        if self.labels is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.labels

    def process_train(self, ds, predicate2id):
        def convert(example):
            events_label = []
            event_list = example.get("event_list", None)
            if event_list is not None:
                for e in event_list:
                    etype = e["event_type"]
                    role = "触发词"
                    argument = e["trigger"]
                    index = e["trigger_start_index"]
                    event = [(predicate2id[etype + "+" + role], index, index + len(argument) - 1)]

                    for a in e["arguments"]:
                        role = a["role"]
                        argument = a["argument"]
                        index = a["argument_start_index"]
                        event.append((predicate2id[etype + "+" + role], index, index + len(argument) - 1))
                    events_label.append(event)
            return {"text": example["text"], "event_list": events_label}

        return ds.map(convert)

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column_name: str,
        label_column_name: str,
        mode: str,
        is_chinese: bool,
    ):

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        tokenized_inputs = tokenizer(
            sentences,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )

        if mode == "train":
            labels = []
            for i, event_list in enumerate(examples[label_column_name]):
                events_label = []
                for e in event_list:
                    event = []
                    for p, h, t in e:
                        try:
                            h = tokenized_inputs.char_to_token(i, h)
                            t = tokenized_inputs.char_to_token(i, t)
                        except Exception:
                            continue
                        if h is None or t is None:
                            continue
                        event.append([p, h, t])
                    events_label.append(event)
                labels.append(events_label)
            tokenized_inputs["labels"] = labels

        return tokenized_inputs

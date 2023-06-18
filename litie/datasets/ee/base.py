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
            predicate2id=self.predicate_to_id,
            mode=mode,
            is_chinese=self.is_chinese,
        )
        return convert_to_features

    @staticmethod
    def duee_v1_process(example):
        events = []
        for e in example["event_list"]:
            offset1 = len(e["trigger"]) - len(e["trigger"].lstrip())
            events.append(
                [
                    [
                        e["event_type"],
                        "触发词",
                        e["trigger"],
                        str(e["trigger_start_index"] + offset1),
                        str(e["trigger_start_index"] + offset1 + len(e["trigger"].strip())),
                    ]
                ]
            )
            for a in e["arguments"]:
                offset2 = len(a["argument"]) - len(a["argument"].lstrip())
                events[-1].append(
                    [
                        e["event_type"],
                        a["role"],
                        a["argument"],
                        str(a["argument_start_index"] + offset2),
                        str(a["argument_start_index"] + offset2 + len(a["argument"].strip())),
                    ]
                )
        del example["event_list"]
        return {"target": events}

    def process_data(self, dataset: Union[Dataset, Dict], stage: Optional[str] = None) -> Union[Dataset, Dict]:
        label_column_name, text_column_name = self._setup_input_fields(dataset, stage)
        self._prepare_labels()

        convert_to_features_train = self.get_process_fct(text_column_name, label_column_name, "train")
        convert_to_features_val = self.get_process_fct(text_column_name, label_column_name, "val")

        train_dataset = dataset["train"].map(self.duee_v1_process)
        val_dataset = dataset["validation"].map(self.duee_v1_process)

        train_dataset = train_dataset.map(
            convert_to_features_train,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train datasets",
            new_fingerprint=f"train-{self.train_max_length}-{self.task_name}",
            num_proc=self.num_workers,
        )

        val_dataset = val_dataset.map(
            convert_to_features_val,
            batched=True,
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

        return all_dataset

    def _setup_input_fields(self, dataset, stage):
        split = "train" if stage == "fit" else "validation"
        column_names = dataset[split].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        label_column_name = "target"
        return label_column_name, text_column_name

    def _prepare_labels(self):
        self.labels = sorted(list(set(self.labels)))
        self.predicate_to_id = {l: i for i, l in enumerate(self.labels)}

    @property
    def schemas(self):
        if self.labels is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.labels

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column_name: str,
        label_column_name: str,
        predicate2id: Dict[str, Any],
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
            for b, events in enumerate(examples[label_column_name]):
                argu_labels = [set() for _ in range(len(predicate2id))]
                head_labels, tail_labels = set(), set()
                for event in events:
                    for i1, (event_type1, role1, word1, head1, tail1) in enumerate(event):
                        head1, tail1 = int(head1), int(tail1)
                        tp1 = predicate2id["@".join([event_type1, role1])]
                        try:
                            h1 = tokenized_inputs.char_to_token(b, head1)
                            t1 = tokenized_inputs.char_to_token(b, tail1 - 1)
                        except:
                            continue

                        if h1 is None or t1 is None:
                            continue
                        argu_labels[tp1].add((h1, t1))

                        for i2, (event_type2, role2, word2, head2, tail2) in enumerate(event):
                            head2, tail2 = int(head2), int(tail2)
                            if i2 > i1:
                                try:
                                    h2 = tokenized_inputs.char_to_token(b, head2)
                                    t2 = tokenized_inputs.char_to_token(b, tail2 - 1)
                                except:
                                    continue

                                if h2 is None or t2 is None:
                                    continue

                                hl = (min(h1, h2), max(h1, h2))
                                tl = (min(t1, t2), max(t1, t2))

                                if hl not in head_labels:
                                    head_labels.add(hl)

                                if tl not in tail_labels:
                                    tail_labels.add(tl)

                for label in argu_labels + [head_labels, tail_labels]:
                    if not label:  # 至少要有一个标签
                        label.add((0, 0))  # 如果没有则用0填充

                argu_labels = [list(l) for l in argu_labels]
                head_labels, tail_labels = list(head_labels), list(tail_labels)

                labels.append(
                    {
                        "argu_labels": argu_labels,
                        "head_labels": head_labels,
                        "tail_labels": tail_labels
                    }
                )

            tokenized_inputs["labels"] = labels

        return tokenized_inputs

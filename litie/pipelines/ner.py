import itertools
from collections import defaultdict
from typing import List, Union, Dict, Set, Optional

import numpy as np
import torch

from .base import BasePredictor
from .utils import auto_splitter
from ..datasets.ner.cnn import DataCollatorForCnnNer
from ..datasets.ner.w2ner import DIST_TO_IDX, DataCollatorForW2Ner
from ..nn.ner import AutoNerTaskModel
from ..utils.logger import tqdm, logger


def set2json(labels: Set) -> Dict:
    """ 将实体集合根据实体类型转换为字典
    """
    res = defaultdict(list)
    for _type, _start, _end, _ent in labels:
        dic = {"start": _start, "end": _end, "text": _ent}
        res[_type].append(dic)
    return res


class NerPredictor(BasePredictor):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 64,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]

            batch_inputs = self.tokenizer(
                batch_inputs,
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )

            batch_inputs['texts'] = inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = self._prepare_inputs(batch_inputs)
            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]


class LearNerPredictor(NerPredictor):

    def __init__(self, *args, schema2prompt: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema2prompt = self.model.config.labels
        schema2prompt = schema2prompt or {}
        self.schema2prompt.update(schema2prompt)
        self.model.config.labels = self.schema2prompt

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        label_annotations = list(self.schema2prompt.values())
        label_inputs = self.tokenizer(
            label_annotations,
            padding=True,
            truncation=True,
            max_length=64,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        label_inputs = {f"label_{k}": v for k, v in label_inputs.items()}

        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = self.tokenizer(
                batch_inputs,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            batch_inputs['texts'] = inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = {**batch_inputs, **label_inputs}
            batch_inputs = self._prepare_inputs(batch_inputs)

            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]


class MrcNerPredictor(NerPredictor):

    def __init__(self, *args, schema2prompt: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if schema2prompt is not None:
            self.schema2prompt = schema2prompt
            self.model.config.labels = self.schema2prompt
        else:
            self.schema2prompt = self.model.config.labels

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        return [self.single_sample_predict(sent, max_length, return_dict) for sent in inputs]

    def single_sample_predict(self, inputs: str, max_length: int = 512, return_dict: bool = True):
        infer_inputs = inputs.replace(" ", "-")  # 防止空格导致位置预测偏移

        first_sentences = list(self.schema2prompt.values())
        second_sentences = [infer_inputs] * len(self.schema2prompt)

        batch_inputs = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=max_length,
            padding=True,
            truncation='only_second',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        batch_inputs['texts'] = [inputs] * len(self.schema2prompt)
        batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

        batch_inputs = self._prepare_inputs(batch_inputs)
        outputs = self.model(**batch_inputs)['predictions']

        return set2json(outputs[0]) if return_dict else outputs[0]


class W2NerPredictor(NerPredictor):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        collate_fn = DataCollatorForW2Ner()
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = [self._process(example, max_length) for example in batch_inputs]

            batch_inputs = collate_fn(batch_inputs)
            batch_inputs = self._prepare_inputs(batch_inputs)

            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]

    def _process(self, text, max_length):
        tokens = [self.tokenizer.tokenize(word) for word in text[:max_length - 2]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _input_ids = self.tokenizer.convert_tokens_to_ids(pieces)
        _input_ids = np.array([self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id])

        length = len(tokens)
        _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
        if self.tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        _dist_inputs = np.zeros((length, length), dtype=np.int)
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i, j in itertools.product(range(length), range(length)):
            _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                _dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19

        _grid_mask = np.ones((length, length), dtype=np.bool)
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]

        encoded_inputs = {k: list(v) for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask])}
        encoded_inputs["text"] = text

        return encoded_inputs


class CnnNerPredictor(NerPredictor):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        collate_fn = DataCollatorForCnnNer()
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = [self._process(example, max_length) for example in batch_inputs]

            batch_inputs = collate_fn(batch_inputs)
            batch_inputs = self._prepare_inputs(batch_inputs)

            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]

    def _process(self, text, max_length):
        _indexes = []
        _bpes = []
        for idx, word in enumerate(text):
            __bpes = self.tokenizer.encode(word, add_special_tokens=False)
            _indexes.extend([idx] * len(__bpes))
            _bpes.extend(__bpes)

        indexes = [0] + [i + 1 for i in _indexes]
        bpes = [self.tokenizer.cls_token_id] + _bpes

        if len(bpes) > max_length - 1:
            indexes = indexes[:max_length - 1]
            bpes = bpes[:max_length - 1]

        def get_new_ins(bpes, indexes):
            bpes.append(self.tokenizer.sep_token_id)
            indexes.append(0)
            return bpes, indexes

        bpes, indexes = get_new_ins(bpes, indexes)
        return {"input_ids": bpes, "indexes": indexes, "text": text}


PREDICTOR_MAP = {
    "mrc": MrcNerPredictor,
    "lear": LearNerPredictor,
    "w2ner": W2NerPredictor,
    "cnn": CnnNerPredictor,
}


def get_auto_ner_predictor(
    task_model_name="crf",
    model_type="bert",
    model=None,
    schema2prompt=None,
    **kwargs,
) -> NerPredictor:

    predictor_class = PREDICTOR_MAP.get(task_model_name, NerPredictor)
    if model is None:
        model = AutoNerTaskModel.create(task_model_name, model_type=model_type)

    if task_model_name not in ["mrc", "lear"]:
        return predictor_class(model, **kwargs)

    return predictor_class(model=model, schema2prompt=schema2prompt, **kwargs)


class NerPipeline(object):
    def __init__(
        self,
        task_model_name="crf",
        model_type="bert",
        model=None,
        model_name_or_path=None,
        tokenizer=None,
        device="cpu",
        use_fp16=False,
        max_seq_len=512,
        batch_size=64,
        split_sentence=False,
        schema2prompt=None,
        load_weights=True,
    ) -> None:

        self._model_name = task_model_name
        self._model_type = model_type
        self._model = model
        self._model_name_or_path = model_name_or_path
        self._tokenizer = tokenizer
        self._device = device
        self._use_fp16 = use_fp16
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._schema2prompt = schema2prompt
        self._load_weights = load_weights

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_ner_predictor(
            self._model_name,
            self._model_type,
            model=self._model,
            schema2prompt=self._schema2prompt,
            model_name_or_path=self._model_name_or_path,
            tokenizer=self._tokenizer,
            device=self._device,
            use_fp16=self._use_fp16,
            load_weights=self._load_weights,
        )

    def __call__(self, inputs):

        texts = inputs
        if isinstance(texts, str):
            texts = [texts]

        max_prompt_len = len(max(self._schema2prompt.values())) if (self._schema2prompt is not None) else -1
        max_predict_len = self._max_seq_len - max_prompt_len - 3

        short_input_texts, self.input_mapping = auto_splitter(
            texts, max_predict_len, split_sentence=self._split_sentence
        )

        results = self.inference_backend.predict(
            short_input_texts, batch_size=self._batch_size, max_length=self._max_seq_len, return_dict=False
        )
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)

        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            single_results = {}
            offset = 0
            for i, v in enumerate(vs):
                if i == 0:
                    single_results = short_results[v]
                else:
                    for res in short_results[v]:
                        tmp = res[0], res[1] + offset, res[2] + offset, res[3]
                        single_results.add(tmp)
                offset += len(short_inputs[v])
            single_results = set2json(single_results) if single_results else {}
            concat_results.append(single_results)
        return concat_results

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value

    @property
    def split(self):
        return self._split_sentence

    @split.setter
    def split(self, value):
        self._split_sentence = value


def vote(entities_list: List[dict], threshold=0.9) -> dict:
    """
    实体级别的投票方式
    entities_list: 所有模型预测出的一个文件的实体
    threshold: 大于阈值，模型预测出来的实体才能被选中
    """
    counts_dict = defaultdict(int)
    entities = defaultdict(list)

    for _entities in entities_list:
        for _type in _entities:
            for _ent in _entities[_type]:
                counts_dict[(_type, _ent["start"], _ent["end"], _ent["text"])] += 1

    for key in counts_dict:
        if counts_dict[key] >= (len(entities_list) * threshold):
            prob = counts_dict[key] / len(entities_list)
            dic = {"start": key[1], "end": key[2], "text": key[3], "probability": prob}
            entities[key[0]].append(dic)

    return entities


class EnsembleNerPipeline(object):
    """ 基于投票法预测实体
    """

    def __init__(self, predicators: List[NerPipeline]):
        self.predicators = predicators

    def __call__(self, text: Union[str, List[str]], threshold=0.8) -> Union[dict, List[dict]]:
        if isinstance(text, str):
            text = [text]

        all_results = [predicator(text) for predicator in self.predicators]
        return [vote(list(entities_list), threshold=threshold) for entities_list in zip(*all_results)]

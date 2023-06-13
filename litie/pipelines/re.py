from collections import defaultdict
from typing import List, Union, Dict, Set

import torch

from .base import BasePredictor
from .utils import auto_splitter
from ..nn.re import AutoReTaskModel
from ..utils.logger import tqdm, logger


def set2json(labels: Set) -> Dict:
    """ 将三元组集合根据关系类型转换为字典 """
    res = defaultdict(list)
    for _type, _sub, _obj in labels:
        dic = {"subject": _sub, "object": _obj}
        res[_type].append(dic)
    return res


class RelationExtractionPredictor(BasePredictor):

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


def get_auto_re_predictor(
    task_model_name="casrel",
    model_type="bert",
    model=None,
    **kwargs,
) -> RelationExtractionPredictor:

    if model is None:
        model = AutoReTaskModel.create(task_model_name, model_type=model_type)

    return RelationExtractionPredictor(model=model, **kwargs)


class RelationExtractionPipeline(object):
    def __init__(
        self,
        task_model_name="casrel",
        model_type="bert",
        model=None,
        model_name_or_path=None,
        tokenizer=None,
        device="cpu",
        use_fp16=False,
        max_seq_len=512,
        batch_size=64,
        split_sentence=False,
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
        self._load_weights = load_weights

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_re_predictor(
            self._model_name,
            self._model_type,
            model=self._model,
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

        max_predict_len = self._max_seq_len - 2
        short_input_texts, self.input_mapping = auto_splitter(
            texts, max_predict_len, split_sentence=self._split_sentence
        )

        results = self.inference_backend.predict(
            short_input_texts, batch_size=self._batch_size, max_length=self._max_seq_len, return_dict=False
        )
        results = self._auto_joiner(results, self.input_mapping)

        return results

    def _auto_joiner(self, short_results, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            group_results = [short_results[v] for v in vs if len(short_results[v]) > 0]
            single_results = set2json(set.union(*group_results))
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


def vote(spo_list: List[dict], threshold=0.9) -> dict:
    """
    三元组级别的投票方式
    entities_list: 所有模型预测出的一个文件的三元组
    threshold: 大于阈值，模型预测出来的实体才能被选中
    """
    counts_dict = defaultdict(int)
    res = defaultdict(list)

    for _spoes in spo_list:
        for _type in _spoes:
            for _spo in _spoes[_type]:
                counts_dict[(_type, _spo["subject"], _spo["object"])] += 1

    for key in counts_dict:
        if counts_dict[key] >= (len(spo_list) * threshold):
            prob = counts_dict[key] / len(spo_list)
            dic = {"subject": key[1], "object": key[2], "probability": prob}
            res[key[0]].append(dic)

    return res


class EnsembleRelationExtractionPipeline(object):
    """ 基于投票法预测三元组
    """

    def __init__(self, predicators: List[RelationExtractionPipeline]):
        self.predicators = predicators

    def __call__(self, text: Union[str, List[str]], threshold=0.8) -> Union[dict, List[dict]]:
        if isinstance(text, str):
            text = [text]

        all_results = [predicator(text) for predicator in self.predicators]
        return [vote(list(spo_list), threshold=threshold) for spo_list in zip(*all_results)]

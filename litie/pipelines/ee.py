from typing import List, Union, Any

import torch

from .base import BasePredictor
from ..nn.ee import AutoEventExtractionTaskModel
from ..utils.logger import tqdm, logger


class DedupList(list):
    """ 定义去重的 list """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def isin(event_a, event_b):
    """ 判断event_a是否event_b的一个子集 """
    if event_a['event_type'] != event_b['event_type']:
        return False
    for argu in event_a['arguments']:
        if argu not in event_b['arguments']:
            return False
    return True


def set2json(events):
    event_list = DedupList()
    for event in events:
        final_event = {
            "event_type": event[0][0],
            "arguments": DedupList()
        }
        for argu in event:
            event_type, role = argu[0], argu[1]
            if role != "触发词":
                final_event["arguments"].append(
                    {
                        "role": role,
                        "argument": argu[2]
                    }
                )
        event_list = [
            event for event in event_list
            if not isin(event, final_event)
        ]
        if not any([isin(final_event, event) for event in event_list]):
            event_list.append(final_event)
    return event_list


class EventExtractionPredictor(BasePredictor):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 64,
        max_length: int = 512,
    ) -> Union[List[Any]]:

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

        return [set2json(o) for o in outputs]


def get_auto_ee_predictor(
    task_model_name="gplinker",
    model_type="bert",
    model=None,
    **kwargs,
) -> EventExtractionPredictor:

    if model is None:
        model = AutoEventExtractionTaskModel.create(task_model_name, model_type=model_type)

    return EventExtractionPredictor(model=model, **kwargs)


class EventExtractionPipeline(object):
    def __init__(
        self,
        task_model_name="gplinker",
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
        self.inference_backend = get_auto_ee_predictor(
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

        results = self.inference_backend.predict(
            texts, batch_size=self._batch_size, max_length=self._max_seq_len,
        )

        return results

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

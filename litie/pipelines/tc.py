from collections import Counter
from typing import List, Union

import numpy as np
import torch

from .base import BasePredictor
from ..nn.tc import AutoTextClassificationTaskModel
from ..utils.logger import tqdm, logger


class TextClassificationPredictor(BasePredictor):

    @torch.no_grad()
    def predict(
        self,
        text_a: Union[str, List[str]],
        text_b: Union[str, List[str]] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> Union[dict, List[dict]]:

        if isinstance(text_a, str):
            text_a = [text_a]
            if text_b is not None and isinstance(text_b, str):
                text_b = [text_b]

        output_list = []
        total_batch = len(text_a) // batch_size + (1 if len(text_a) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text_a = text_a[batch_id * batch_size: (batch_id + 1) * batch_size]
            if text_b is not None:
                batch_text_b = text_b[batch_id * batch_size: (batch_id + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_text_a,
                    batch_text_b,
                    max_length=max_length,
                    padding=True,
                    truncation='only_second',
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(
                    batch_text_a,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

            inputs = self._prepare_inputs(inputs)
            outputs = self.model(**inputs)

            outputs = np.asarray(outputs['logits'].cpu()).argmax(-1)
            output_list.extend(outputs)

        if hasattr(self.model.config, "tc_label2id"):
            self.id2label = {int(v): k for k, v in self.model.config.tc_label2id.items()}
            output_list = [self.id2label[o] for o in output_list]

        return output_list


def get_auto_tc_predictor(
    task_model_name="tc",
    model_type="bert",
    model=None,
    **kwargs,
) -> TextClassificationPredictor:

    if model is None:
        model = AutoTextClassificationTaskModel.create(task_model_name, model_type=model_type)

    return TextClassificationPredictor(model=model, **kwargs)


class TextClassificationPipeline(object):
    def __init__(
        self,
        task_model_name="tc",
        model_type="bert",
        model=None,
        model_name_or_path=None,
        tokenizer=None,
        device="cpu",
        use_fp16=False,
        max_seq_len=512,
        batch_size=64,
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
        self._load_weights = load_weights

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_tc_predictor(
            self._model_name,
            self._model_type,
            model=self._model,
            model_name_or_path=self._model_name_or_path,
            tokenizer=self._tokenizer,
            device=self._device,
            use_fp16=self._use_fp16,
            load_weights=self._load_weights,
        )

    def __call__(self, text_a: Union[str, List[str]], text_b: Union[str, List[str]] = None):
        return self.inference_backend.predict(
            text_a,
            text_b,
            batch_size=self._batch_size,
            max_length=self._max_seq_len
        )

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value


class EnsembleTextClassificationPipeline(object):
    def __init__(self, predicators: List[TextClassificationPipeline]):
        self.predicators = predicators

    def __call__(self, text_a: Union[str, List[str]], text_b: Union[str, List[str]] = None, threshold=0.8):
        all_results = [predicator(text_a, text_b) for predicator in self.predicators]
        return [Counter(list(label_list)).most_common(1)[0][0] for label_list in zip(*all_results)]

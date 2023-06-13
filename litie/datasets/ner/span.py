from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import NerDataModule
from ..utils import batchify_ner_labels


@dataclass
class DataCollatorForSpanNer:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in self.ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            return batchify_ner_labels(batch, features, return_offset_mapping=True)

        batch_start_positions = torch.zeros_like(batch["input_ids"])
        batch_end_positions = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_start_positions[i, start] = tag + 1
                batch_end_positions[i, end] = tag + 1

        batch['start_positions'] = batch_start_positions
        batch['end_positions'] = batch_end_positions

        return batch


class SpanForNerDataModule(NerDataModule):

    config_name: str = "span"

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForSpanNer(tokenizer=self.tokenizer, ignore_list=ignore_list)

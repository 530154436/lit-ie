from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import NerDataModule
from ..utils import batchify_ner_labels


@dataclass
class DataCollatorForTPLinkerPlusNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
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

        bs, seqlen = batch["input_ids"].shape
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, self.num_labels, dtype=torch.long)

        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_shaking_tag[i, start, end, tag] = 1

        batch["labels"] = batch_shaking_tag.masked_select(mask[None, :, :, None]).reshape(bs, -1, self.num_labels)

        return batch


class TPlinkerNerDataModule(NerDataModule):

    config_name: str = "tplinker"

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForTPLinkerPlusNer(
            tokenizer=self.tokenizer,
            num_labels=len(self.labels),
            ignore_list=ignore_list,
        )

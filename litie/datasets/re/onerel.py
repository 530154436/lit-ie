from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import RelationExtractionDataModule
from ..utils import batchify_re_labels


@dataclass
class DataCollatorForOneRel:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
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
            return batchify_re_labels(batch, features, return_offset_mapping=True)

        bs, seqlen = batch["input_ids"].shape
        seqlens = batch["attention_mask"]
        batch_labels = torch.zeros(bs, self.num_predicates, seqlen, seqlen, dtype=torch.int32)

        for i, lb in enumerate(labels):
            l = seqlens[i].sum()
            for sh, st, p, oh, ot in lb:
                batch_labels[i, p, sh, oh] = 1
                batch_labels[i, p, sh, ot] = 2
                batch_labels[i, p, st, ot] = 3

            batch_labels[i, :, l:, l:] = -100

        batch["labels"] = labels

        return batch


class OneRelForReDataModule(RelationExtractionDataModule):

    config_name: str = "onerel"

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForOneRel(
            tokenizer=self.tokenizer,
            num_predicates=len(self.labels),
            ignore_list=ignore_list,
        )

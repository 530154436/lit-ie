from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import EventExtractionDataModule
from ..utils import batchify_ee_labels, sequence_padding


@dataclass
class DataCollatorForGPLinker:

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
            return batchify_ee_labels(batch, features, return_offset_mapping=True)

        batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
        for lb in labels:
            batch_argu_labels.append(sequence_padding(lb["argu_labels"]))
            batch_head_labels.append(sequence_padding(lb["head_labels"]))
            batch_tail_labels.append(sequence_padding(lb["tail_labels"]))

        batch["argu_labels"] = torch.from_numpy(sequence_padding(batch_argu_labels, seq_dims=2))
        batch["head_labels"] = torch.from_numpy(sequence_padding(batch_head_labels, seq_dims=2))
        batch["tail_labels"] = torch.from_numpy(sequence_padding(batch_tail_labels, seq_dims=2))

        return batch


class GPLinkerForEeDataModule(EventExtractionDataModule):

    config_name: str = "gplinker"

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target", "id"]
        return DataCollatorForGPLinker(
            tokenizer=self.tokenizer,
            num_predicates=len(self.labels),
            ignore_list=ignore_list,
        )

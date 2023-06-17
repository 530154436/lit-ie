from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import EventExtractionDataModule
from ..utils import batchify_re_labels


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
            return batchify_re_labels(batch, features, return_offset_mapping=True)

        bs = batch["input_ids"].size(0)
        max_head_num = max([len(lb["head_labels"]) for lb in labels])
        max_tail_num = max([len(lb["tail_labels"]) for lb in labels])
        max_argu_num = max([(len(lb) - 1) // 2 for label in labels for lb in label["argu_labels"]])

        batch_argu_labels = torch.zeros(bs, self.num_predicates, max_argu_num * 2, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, 1, max_head_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, 1, max_tail_num, 2, dtype=torch.long)

        for b, lb in enumerate(labels):
            # argu_labels
            for argu in lb["argu_labels"]:
                batch_argu_labels[b, argu[0], : len(argu[1:])] = torch.tensor(argu[1:], dtype=torch.long)

            # head_labels
            for ih, (h1, h2) in enumerate(lb["head_labels"]):
                batch_head_labels[b, 0, ih, :] = torch.tensor([h1, h2], dtype=torch.long)

            # tail_labels
            for it, (t1, t2) in enumerate(lb["tail_labels"]):
                batch_tail_labels[b, 0, it, :] = torch.tensor([t1, t2], dtype=torch.long)

        batch["argu_labels"] = batch_argu_labels.reshape(bs, self.num_predicates, max_argu_num, 2)
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels

        return batch


class GPLinkerForEeDataModule(EventExtractionDataModule):

    config_name: str = "gplinker"

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForGPLinker(
            tokenizer=self.tokenizer,
            num_predicates=len(self.labels),
            ignore_list=ignore_list,
        )

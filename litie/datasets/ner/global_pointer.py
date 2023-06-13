from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from .base import NerDataModule
from ..utils import sequence_padding, batchify_ner_labels


@dataclass
class DataCollatorForGlobalPointerNer:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    is_sparse: Optional[bool] = False
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
        if self.is_sparse:
            batch_labels = []
            for lb in labels:
                label = [set() for _ in range(self.num_labels)]
                for start, end, tag in lb:
                    label[tag].add((start, end))
                for l in label:
                    if not l:  # 至少要有一个标签
                        l.add((0, 0))  # 如果没有则用0填充
                label = sequence_padding([list(l) for l in label])
                batch_labels.append(label)
            batch_labels = torch.from_numpy(sequence_padding(batch_labels, seq_dims=2))
        else:
            batch_labels = torch.zeros(bs, self.num_labels, seqlen, seqlen, dtype=torch.long)
            for i, lb in enumerate(labels):
                for start, end, tag in lb:
                    batch_labels[i, tag, start, end] = 1  # 0为"O"

        batch["labels"] = batch_labels

        return batch


class GlobalPointerForNerDataModule(NerDataModule):

    config_name: str = "global_pointer"

    def __init__(self, *args, sparse: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sparse = sparse

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForGlobalPointerNer(
            tokenizer=self.tokenizer,
            num_labels=len(self.labels),
            is_sparse=self.sparse,
            ignore_list=ignore_list,
        )

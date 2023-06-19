from itertools import groupby
from typing import Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..model_utils import RelationExtractionOutput, MODEL_MAP
from ...datasets.utils import tensor_to_numpy
from ...layers.global_pointer import EfficientGlobalPointer
from ...losses import SparseMultilabelCategoricalCrossentropy


class DedupList(list):
    """ 定义去重的 list """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """ 构建邻集（host节点与其所有邻居的集合） """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _,  h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def get_auto_gplinker_ee_model(
    model_type: Optional[str] = "bert",
    base_model: Optional[PreTrainedModel] = None,
    parent_model: Optional[PreTrainedModel] = None,
):
    if base_model is None and parent_model is None:
        base_model, parent_model = MODEL_MAP[model_type]

    class GPLinker(parent_model):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, self.base_model_prefix, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.hidden_size = config.hidden_size
            # 实体首尾对应，需要相对位置编码且保证首不超过尾
            self.argu_tagger = EfficientGlobalPointer(config.hidden_size, config.num_predicates, config.head_size)
            self.head_tagger = EfficientGlobalPointer(
                config.hidden_size,
                1,
                config.head_size,
                use_rope=False,
            )
            self.tail_tagger = EfficientGlobalPointer(
                config.hidden_size,
                1,
                config.head_size,
                use_rope=False,
            )

            self.has_trigger = getattr(config, "trigger", True)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            argu_labels: Optional[torch.Tensor] = None,
            head_labels: Optional[torch.Tensor] = None,
            tail_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            sequence_output = self.dropout(outputs[0])  # [batch_size, seq_len, hidden_size]

            # [batch_size, 2, seq_len, seq_len]
            argu_logits = self.argu_tagger(sequence_output, mask=attention_mask)
            # [batch_size, num_predicates, seq_len, seq_len]
            head_logits = self.head_tagger(sequence_output, mask=attention_mask)
            # [batch_size, num_predicates, seq_len, seq_len]
            tail_logits = self.tail_tagger(sequence_output, mask=attention_mask)

            loss, predictions = None, None
            if argu_labels is not None and head_labels is not None and tail_labels is not None:
                argu_loss = self.compute_loss([argu_logits, argu_labels])
                head_loss = self.compute_loss([head_logits, head_labels])
                tail_loss = self.compute_loss([tail_logits, tail_labels])
                loss = (argu_loss + head_loss + tail_loss) / 3

            if not self.training:
                predictions = self.decode(
                    argu_logits, head_logits, tail_logits, attention_mask, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, argu_logits, head_logits, tail_logits, masks, texts, offset_mapping):
            all_event_list = []
            batch_size = argu_logits.shape[0]
            masks = tensor_to_numpy(masks)

            entity_logits = tensor_to_numpy(argu_logits)
            head_logits = tensor_to_numpy(head_logits)
            tail_logits = tensor_to_numpy(tail_logits)
            decode_thresh = getattr(self.config, "decode_thresh", 0.0)

            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}
            for bs in range(batch_size):
                l = masks[bs].sum()
                text, mapping = texts[bs], offset_mapping[bs]

                # 抽取论元
                argus = set()
                _entity_logits = entity_logits[bs]
                for p, h, t in zip(*np.where(_entity_logits > decode_thresh)):
                    if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                        continue
                    p = id2predicate[p].rsplit('@', 1)
                    argus.add((*p, h, t))

                # 构建链接
                links = set()
                _head_logits, _tail_logits = head_logits[bs], tail_logits[bs]
                for i1, (_, _, h1, t1) in enumerate(argus):
                    for i2, (_, _, h2, t2) in enumerate(argus):
                        if i2 > i1:
                            if _head_logits[0, min(h1, h2), max(h1, h2)] > decode_thresh and _tail_logits[
                                    0, min(t1, t2), max(t1, t2)] > decode_thresh:
                                links.add((h1, t1, h2, t2))
                                links.add((h2, t2, h1, t1))
                # 析出事件
                events = []
                for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
                    for event in clique_search(list(sub_argus), links):
                        events.append([])
                        for argu in event:
                            start, end = mapping[argu[2]][0], mapping[argu[3]][1]
                            events[-1].append(
                                (
                                    argu[0],
                                    argu[1],
                                    text[start: end],
                                    start,
                                    end
                                )
                            )
                        if self.has_trigger and all([argu[1] != "触发词" for argu in event]):
                            events.pop()

                all_event_list.append(events)

            return all_event_list

        def compute_loss(self, inputs):
            preds, target = inputs[:2]
            shape = preds.shape
            target = target[..., 0] * shape[2] + target[..., 1]  # [bsz, heads, num_spoes]
            preds = preds.reshape(shape[0], -1, np.prod(shape[2:]))
            loss_fct = SparseMultilabelCategoricalCrossentropy(mask_zero=True)
            return loss_fct(preds, target.long()).sum(dim=1).mean()

    return GPLinker


def get_gplinker_ee_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}
    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id, "head_size": 64,
    }
    model_config.update(kwargs)
    return model_config

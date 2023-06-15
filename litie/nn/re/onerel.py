from typing import Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..model_utils import RelationExtractionOutput, MODEL_MAP
from ...datasets.utils import tensor_to_numpy


def get_auto_onerel_re_model(
    model_type: Optional[str] = "bert",
    base_model: Optional[PreTrainedModel] = None,
    parent_model: Optional[PreTrainedModel] = None,
):
    if base_model is None and parent_model is None:
        base_model, parent_model = MODEL_MAP[model_type]

    class OneRel(parent_model):
        """
        基于`BERT`的`GPLinker`关系抽取模型
        + 📖 模型的整体思路将三元组抽取分解为实体首尾对应、主体-客体首首对应、主体-客体尾尾对应
        + 📖 通过采用类似多头注意力得分计算的机制将上述三种关系最后映射到一个二维矩阵
        + 📖 每种关系都采用`GlobalPointer`来建模

        Args:
            `config`: 模型的配置

        Reference:
            ⭐️ [OneRel：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888)
            🚀 [Code](https://github.com/ssnvxia/OneRel)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, self.base_model_prefix, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.tag_size = 4
            self.projection_matrix_layer = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 3)
            self.relation_matrix_layer = nn.Linear(self.config.hidden_size * 3, self.config.num_predicates * self.tag_size)
            self.entity_pair_dropout = nn.Dropout(getattr(config, "entity_pair_dropout", 0.1))

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
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

            # encoded_text: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = sequence_output.size()
            # head: [batch_size, seq_len * seq_len, hidden_size]
            head_representation = sequence_output.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size).reshape(
                batch_size, seq_len * seq_len, hidden_size)
            # tail: [batch_size, seq_len * seq_len, hidden_size]
            tail_representation = sequence_output.repeat(1, seq_len, 1)
            # [batch_size, seq_len * seq_len, hidden_size * 2]
            entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

            # [batch_size, seq_len * seq_len, hidden_size * 3]
            entity_pairs = self.projection_matrix_layer(entity_pairs)
            entity_pairs = self.entity_pair_dropout(entity_pairs)

            # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
            triple_scores = self.relation_matrix_layer(entity_pairs).reshape(
                batch_size, seq_len, seq_len, self.config.num_predicates, self.tag_size)

            loss, predictions = None, None
            if labels is not None:
                loss = self.compute_loss([triple_scores, labels])

            if not self.training:
                predictions = self.decode(
                    triple_scores, attention_mask, texts, offset_mapping
                )

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, logits, masks, texts, offset_mapping):
            all_spo_list = []
            batch_size = logits.shape[0]
            masks = tensor_to_numpy(masks)
            logits = tensor_to_numpy(torch.argmax(logits, dim=-1).permute(0, 3, 1, 2))

            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}
            for bs in range(batch_size):
                _logits = logits[bs]
                l = masks[bs].sum()
                text, mapping = texts[bs], offset_mapping[bs]

                hs, hts, ts = {}, {}, {}
                for obj, tag in [(hs, 1), (hts, 2), (ts, 3)]:
                    for p, h, t in zip(*np.where(_logits == filter)):
                        if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                            continue
                        if p not in obj:
                            obj[p] = []
                        obj[p].append((h, t))

                spoes = set()
                for p in hs.keys() & ts.keys() & hts.keys():
                    ht_list = hts[p]
                    for sh, oh in hs[p]:
                        for st, ot in ts[p]:
                            if sh <= st and oh <= ot:
                                if (sh, ot) in ht_list:
                                    spoes.add((
                                        id2predicate[p],
                                        text[mapping[sh][0]: mapping[st][1]],
                                        text[mapping[oh][0]: mapping[ot][1]]
                                    ))
                all_spo_list.append(spoes)
            return all_spo_list

        def compute_loss(self, inputs):
            logits, labels = inputs[:2]
            loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss = loss_fn(logits.permute(0, 4, 3, 1, 2), labels)
            return loss.mean(-1).mean(-1).sum() / logits.size(0)

    return OneRel


def get_onerel_re_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}
    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id,
    }
    model_config.update(kwargs)
    return model_config

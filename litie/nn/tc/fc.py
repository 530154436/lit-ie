from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from ..model_utils import MODEL_MAP
from ...layers.dropouts import MultiSampleDropout
from ...layers.pooling import Pooler
from ...losses import RDropLoss, FocalLoss, LabelSmoothingCrossEntropy


def get_auto_fc_tc_model(
    model_type: Optional[str] = "bert",
    base_model: Optional[PreTrainedModel] = None,
    parent_model: Optional[PreTrainedModel] = None,
):
    if base_model is None and parent_model is None:
        base_model, parent_model = MODEL_MAP[model_type]

    class SequenceClassification(parent_model):
        """
        基于BERT的文本分类模型

        Args:
            config: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.num_labels = config.num_labels
            self.pooler_type = getattr(config, "pooler_type", "cls")

            if self.pooler_type != "cls":
                self.config.output_hidden_states = True

            setattr(self, self.base_model_prefix, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )

            self.dropout = nn.Dropout(classifier_dropout)
            self.pooling = Pooler(self.pooler_type)

            self.use_mdp = getattr(config, "use_mdp", False)
            if self.use_mdp:
                self.classifier = MultiSampleDropout(
                    config.hidden_size,
                    config.num_labels,
                    K=getattr(config, "k", 3),
                    p=getattr(config, "p", 0.5),
                )
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> SequenceClassifierOutput:

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            pooled_output = self.dropout(self.pooling(outputs, attention_mask))
            logits = self.classifier(pooled_output)

            loss = self.compute_loss([logits, labels]) if labels is not None else None
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def compute_loss(self, inputs):
            logits, labels = inputs[:2]
            loss_type = getattr(self.config, "loss_type", "cross_entropy")
            if loss_type == "r-drop":
                alpha = getattr(self.config, "alpha", 4)
                loss_fct = RDropLoss(alpha=alpha, rank="updown")
            elif loss_type == "focal":
                loss_fct = FocalLoss()
            elif loss_type == "label-smoothing":
                loss_fct = LabelSmoothingCrossEntropy()
            else:
                loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    return SequenceClassification


def get_fc_tc_model_config(label2id, **kwargs):
    model_config = {
        "num_labels": len(label2id),
        "pooler_type": "cls",
        "classifier_dropout": 0.3,
        "tc_label2id": label2id,
    }
    model_config.update(kwargs)
    return model_config

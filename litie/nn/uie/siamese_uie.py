from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder

from ..model_utils import UIEModelOutput, MODEL_MAP
from ...losses import MultilabelCategoricalCrossentropy


def get_auto_siamese_uie_model(
    model_type: Optional[str] = "bert",
    base_model: Optional[PreTrainedModel] = None,
    parent_model: Optional[PreTrainedModel] = None,
):
    if base_model is None and parent_model is None:
        base_model, parent_model = MODEL_MAP[model_type]

    class SiameseUIE(parent_model):
        """
        `SiameseUIE` 基于递归的训练推理架构，不仅可以实现常见的 `NER`、`RE`、`EE`、`ABSA` 这类包含一个或两个抽取片段的信息抽取任务，
        也可以实现包含更多抽取片段的信息抽取任务

        + 📖 基于孪生神经网络的思想，将预训练语言模型（`PLM`）的前 `N-n` 层改为双流，后 `n` 层改为单流。
        + 📖 语言模型的底层更多的是实现局部的简单语义信息的交互，顶层更多的是深层信息的交互，因此前 `N-n` 层不让 `Prompt` 和 `Text` 做过多的交互，而是分别单独编码
        + 📖 将前 `N-n` 层 `Text` 的隐向量表示缓存下来，实现将推理速度提升 30%

        Args:
            `config`: 模型的配置对象

        Reference:
            ⭐️ [SiameseUIE](https://zhuanlan.zhihu.com/p/634138767)
            🚀 [Code](https://github.com/modelscope/modelscope/blob/master/modelscope/models/nlp/bert/siamese_uie.py)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, self.base_model_prefix, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.head = nn.Linear(config.hidden_size, 1)
            self.tail = nn.Linear(config.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

            # Initialize weights and apply final processing
            self.set_crossattention_layer(getattr(config, "num_cross_attention_layers", 6))

        def set_crossattention_layer(self, num_hidden_layers=6):
            crossattention_config = deepcopy(self.config)
            crossattention_config.num_hidden_layers = num_hidden_layers
            self.config.num_hidden_layers -= num_hidden_layers
            self.crossattention = BertEncoder(crossattention_config)
            self.crossattention.layer = self.backbone.encoder.layer[self.config.num_hidden_layers:]
            self.backbone.encoder.layer = self.backbone.encoder.layer[:self.config.num_hidden_layers]

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            prompt_input_ids: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
        ) -> UIEModelOutput:

            # text states
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            sequence_output = self.dropout(outputs[0])

            # prompt states
            position_ids = torch.arange(prompt_input_ids.size(1)).expand(1, -1) + input_ids.size(1).to(input_ids.device)
            prompt_sequence_output = getattr(self, self.base_model_prefix)(
                prompt_input_ids,
                attention_mask=cross_attention_mask,
                token_type_ids=torch.ones_like(cross_attention_mask),
                position_ids=position_ids,
            )

            # fusion states
            sequence_output = torch.cat([sequence_output, prompt_sequence_output], dim=1)
            cat_attention_mask = torch.cat([attention_mask, cross_attention_mask], dim=1)
            cat_attention_mask = self.backbone.get_extended_attention_mask(
                cat_attention_mask,
                sequence_output.size()[:2]
            )
            sequence_output = self.crossattention(
                hidden_states=sequence_output, attention_mask=cat_attention_mask
            )[0][:, :input_ids.size()[1], :]

            start_logits = self.head(sequence_output).squeeze(-1)
            end_logits = self.tail(sequence_output).squeeze(-1)

            start_prob = self.sigmoid(start_logits)
            end_prob = self.sigmoid(end_logits)

            loss, predictions = None, None
            if start_positions is not None and end_positions is not None:
                head_loss = self.compute_loss([start_logits, start_positions])
                tail_loss = self.compute_loss([end_logits, end_positions])
                loss = (head_loss + tail_loss) / 2.0

            return UIEModelOutput(
                loss=loss,
                start_prob=start_prob,
                end_prob=end_prob,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def compute_loss(self, inputs):
            preds, target = inputs[:2]
            batch_size = preds.shape[0]
            loss_fct = MultilabelCategoricalCrossentropy()
            return loss_fct(preds.reshape(batch_size, -1), target.reshape(batch_size, -1))

        @property
        def backbone(self):
            return getattr(self, self.base_model_prefix)

    return SiameseUIE


def get_siamese_uie_model_config(**kwargs):
    model_config = {"num_cross_attention_layers": 6}
    model_config.update(kwargs)
    return model_config

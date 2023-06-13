import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput


def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        extended_attention_mask = attention_mask

    return (1.0 - extended_attention_mask) * -10000.0


class TransformerEncoderLayer(nn.Module):
    """
    Args:
        config.hidden_size: 隐藏层维度
        config.num_attention_heads: 注意力head数
        config.attention_probs_dropout_prob: dropout
        position_embedding_type=None: 位置编码类型
        config.is_decoder: 是否为解码器

        config.intermediate_size: 中间层维度
        config.hidden_act: 中间层激活函数
        config.hidden_dropout_prob: 中间层dropout
    """

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask)
        attention_outputs = self.attention(hidden_states, attention_mask=attention_mask)
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_outputs = self.intermediate(attention_output)
        layer_output = self.output(intermediate_outputs, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask=None
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)

        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return (layer_output,) + outputs

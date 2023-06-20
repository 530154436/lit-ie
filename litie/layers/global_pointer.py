import math

import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .position import (
    SinusoidalPositionEncoding,
    RoPEPositionEncoding,
    RelativePositionsEncoding
)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(self, hidden_size, heads, head_size, use_rope=True, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.use_rope:
            self.position_embedding = RoPEPositionEncoding(head_size)

    def forward(self, inputs, mask=None):
        """
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.dense(inputs)  # [..., heads*head_size*2]
        sequence_output = torch.stack(
            torch.chunk(sequence_output, self.heads, dim=-1), dim=-2
        )  # [..., heads, head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.use_rope:
            # 为了seq_len维度在-2, 所以进行了转置
            qw = self.position_embedding(qw.transpose(1, -2)).transpose(1, -2)
            kw = self.position_embedding(kw.transpose(1, -2)).transpose(1, -2)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """

    def __init__(self, hidden_size, heads, head_size, use_rope=True, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.use_rope:
            self.position_embedding = RoPEPositionEncoding(head_size)

    def forward(self, inputs, mask=None):
        """
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.use_rope:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(
            torch.chunk(bias_input, self.heads, dim=-1), dim=-2
        ).transpose(1, 2) / 2  # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class Biaffine(nn.Module):
    """双仿射网络 Named Entity Recognition as Dependency Parsing
    """

    def __init__(self, hidden_size, head_size, num_labels, max_len=512,
                 add_position=True, bias=True, tri_mask=True):
        super(Biaffine, self).__init__()
        # use absolute position encoding
        self.add_position = add_position
        self.bias = bias
        self.tri_mask = tri_mask
        if add_position:
            self.position_encoding = SinusoidalPositionEncoding(max_len, hidden_size)

        # add lstm layers
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size,
                                  num_layers=1,
                                  batch_first=True,
                                  dropout=0.2,
                                  bidirectional=True)

        # 头预测层
        self.start_layer = nn.Sequential(nn.Linear(2 * hidden_size, head_size), nn.ReLU())
        # 尾预测层
        self.end_layer = nn.Sequential(nn.Linear(2 * hidden_size, head_size), nn.ReLU())

        self.u = nn.Parameter(torch.Tensor(head_size + self.bias, num_labels, head_size + self.bias))
        self.w = nn.Parameter(torch.Tensor(2 * (head_size + self.bias) + 1, num_labels))

        self.init_weights()

    def init_weights(self):
        """ 参数初始化 """
        torch.nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, hidden_state, mask=None):
        # hidden_state = [batch size, seq_len, hidden_size]
        seq_len = hidden_state.shape[1]

        if self.add_position:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand_as(hidden_state)
            hidden_state += self.position_encoding(position_ids)

        hidden_state, _ = self.lstm(hidden_state)
        start_logits = self.start_layer(hidden_state)  # [b, l, d]
        end_logits = self.end_layer(hidden_state)

        if self.bias:
            # [b, l, d + 1]
            start_logits = torch.cat((start_logits, torch.ones_like(start_logits[..., :1])), dim=-1)
            end_logits = torch.cat((end_logits, torch.ones_like(end_logits[..., :1])), dim=-1)

        start_logits_con = torch.unsqueeze(start_logits, 1)  # [b, 1, l, d + 1]
        end_logits_con = torch.unsqueeze(end_logits, 2)  # [b, l, 1, d + 1]

        start_logits_con = start_logits_con.repeat(1, seq_len, 1, 1)  # [b, l, l, d + 1]
        end_logits_con = end_logits_con.repeat(1, 1, seq_len, 1)

        # [b, l, l, 2(d + 1)]
        concat_start_end = torch.cat([start_logits_con, end_logits_con], dim=-1)
        # [b, l, l, 2(d + 1) + 1]
        concat_start_end = torch.cat([concat_start_end, torch.ones_like(concat_start_end[..., :1])], dim=-1)

        # bix, xny, bjy -> bijn: [b, l, l, n]
        logits_1 = torch.einsum('bix, xny, bjy -> bijn', start_logits, self.u, end_logits)
        logits_2 = torch.einsum('bijy, yn -> bijn', concat_start_end, self.w)

        logits = logits_1 + logits_2
        logits = logits.permute(0, 3, 1, 2)  # [b, n, l, l]

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class UnlabeledEntity(nn.Module):
    """ https://arxiv.org/pdf/2012.05426.pdf
    """

    def __init__(self, hidden_size, num_labels, max_len=512, position_type='relative',
                 tri_mask=True):  # [None, 'absolute', 'relative']
        super(UnlabeledEntity, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.position_type = position_type
        self.tri_mask = tri_mask

        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size, self.num_labels)

        # 位置编码
        if self.position_type == 'absolute':
            self.position_encoding = SinusoidalPositionEncoding(max_len, hidden_size)
        elif self.position_type == 'relative':
            self.relative_positions_encoding = RelativePositionsEncoding(
                qlen=max_len,
                klen=max_len,
                embedding_size=hidden_size * 4
            )

    def forward(self, hidden_state, mask=None):
        seq_len = hidden_state.shape[1]

        # 绝对位置编码
        if self.position_type == 'absolute':
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand_as(hidden_state)
            hidden_state += self.position_encoding(position_ids)

        start_logits = torch.unsqueeze(hidden_state, 1)  # [b, 1, l, h]
        end_logits = torch.unsqueeze(hidden_state, 2)  # [b, l, 1, h]

        start_logits = start_logits.repeat(1, seq_len, 1, 1)  # [b, l, l, h]
        end_logits = end_logits.repeat(1, 1, seq_len, 1)

        concat_inputs = torch.cat([end_logits, start_logits,
                                   end_logits - start_logits,
                                   end_logits.mul(start_logits)], dim=-1)

        if self.position_type == 'relative':
            relations_keys = self.relative_positions_encoding(seq_len, seq_len)
            concat_inputs += relations_keys

        hij = torch.tanh(self.Wh(concat_inputs))
        logits = self.Wo(hij)  # [b, l, l, n]

        logits = logits.permute(0, 3, 1, 2)  # [b, n, l, l]

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type="lstm"):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])],
                dim=1
            )
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        """
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid):
    """Returns: [seq_len, d_hid]
    """
    embeddings_table = torch.zeros(n_position, d_hid)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))

    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)

    return embeddings_table


class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=True)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class RelativePositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """
    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(RelativePositionsEncoding, self).__init__()
        # 生成相对位置矩阵
        vocab_size = max_relative_position * 2 + 1
        distance_mat = torch.arange(klen)[None, :] - torch.arange(qlen)[:, None]  # 列数-行数, [query_len, key_len]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        # sinusoid_encoding编码的位置矩阵
        embeddings_table = get_sinusoid_encoding_table(vocab_size, embedding_size)
        position_embeddings = nn.Embedding.from_pretrained(embeddings_table, freeze=True)(final_mat)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, qlen, klen):
        return self.position_embeddings[:qlen, :klen, :]


class T5RelativePositionsEncoding(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """
    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(T5RelativePositionsEncoding, self).__init__()
        # 生成相对位置矩阵
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        relative_position = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not is_decoder,
            num_buckets=relative_attention_num_buckets,
        )
        self.register_buffer('relative_position', relative_position)

    def forward(self, qlen, klen):
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """直接来源于transformer
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)

        return ret


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """

    def __init__(self, embedding_size, rope_rank='adjacent', **kwargs):
        super(RoPEPositionEncoding, self).__init__()
        self.max_seq_len_cache = -1
        self.embedding_size = embedding_size
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列, 目前只在chatglm中看到updown排列
        assert rope_rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rope_rank = rope_rank

    def initialize(self, max_position):
        position_embeddings = get_sinusoid_encoding_table(max_position, self.embedding_size)  # [seq_len, hdsz]
        if self.rope_rank == 'adjacent':
            cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
            sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)  # [seq_len, hdsz]
        elif self.rope_rank == 'updown':  # 目前仅chatglm使用
            cos_position = position_embeddings[:, 1::2].repeat(1, 2)  # [seq_len, hdsz]
            sin_position = position_embeddings[:, ::2].repeat(1, 2)  # [seq_len, hdsz]
        else:
            raise ValueError('Args `rope_rank` only support `adjacent` and `adjacent` mode')
        return cos_position, sin_position

    def forward(self, qw, position_ids=None, seq_dim=-2):
        # MultiHeadAttentionLayer中qw是[btz, n_heads, seq_len, head_size]
        # GlobalPointer中*转置*后qw是[btz, n_heads, seq_len, head_size]
        # EfficientGlobalPointer中qw是[btz, seq_len, head_size]
        if self.rope_rank == 'adjacent':
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        elif self.rope_rank == 'updown':  # 目前仅chatglm使用
            qw2 = torch.cat(
                [-qw[..., qw.shape[-1] // 2:], qw[..., :qw.shape[-1] // 2]], dim=-1
            )  # cat和stack+reshape是结果不同的

        # 超过缓存长度
        seq_len = position_ids.max() + 1 if position_ids is not None else qw.shape[seq_dim]
        if seq_len > self.max_seq_len_cache:
            cos_position, sin_position = self.initialize(seq_len)
            self.cos_position, self.sin_position = cos_position.type_as(qw).to(qw.device), sin_position.type_as(qw).to(
                qw.device)
            self.max_seq_len_cache = seq_len

        # 传入position_ids来获取cos和sin, 主要是在use_cache时候能直接取到对应位置的编码
        if position_ids is not None:
            cos = F.embedding(position_ids, self.cos_position)
            sin = F.embedding(position_ids, self.sin_position)
            return qw * cos + qw2 * sin

        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]

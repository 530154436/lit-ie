from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple

import torch
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertTokenizer,
    BertModel,
    BertPreTrainedModel,
    BertTokenizerFast,
    ErnieModel,
    ErniePreTrainedModel,
    NezhaModel,
    NezhaPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
    XLNetTokenizer,
)
from transformers.file_utils import ModelOutput

from .chinese_bert import ChineseBertModel, ChineseBertTokenizerFast
from .roformer import RoFormerModel, RoFormerPreTrainedModel

MODEL_MAP = OrderedDict(
    {
        "bert": (BertModel, BertPreTrainedModel),
        "ernie": (ErnieModel, ErniePreTrainedModel),
        "roformer": (RoFormerModel, RoFormerPreTrainedModel),
        "nezha": (NezhaModel, NezhaPreTrainedModel),
        "albert": (AlbertModel, AlbertPreTrainedModel),
        "xlnet": (XLNetModel, XLNetPreTrainedModel),
        "chinese-bert": (ChineseBertModel, BertPreTrainedModel),
    }
)

TOKENIZER_MAP = OrderedDict(
    {
        "bert": BertTokenizerFast,
        "ernie": BertTokenizerFast,
        "roformer": BertTokenizerFast,
        "nezha": BertTokenizerFast,
        "albert": AlbertTokenizer,
        "xlnet": XLNetTokenizer,
        "chinese-bert": ChineseBertTokenizerFast,
    }
)


@dataclass
class SequenceLabelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SpanOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: Optional[torch.FloatTensor] = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class UIEModelOutput(ModelOutput):
    """
    Output class for outputs of UIE.
    losses (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Total spn extraction losses is the sum of a Cross-Entropy for the start and end positions.
    start_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Span-start scores (after Sigmoid).
    end_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Span-end scores (after Sigmoid).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layers, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
        Attention weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    start_positions: torch.FloatTensor = None
    end_positions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

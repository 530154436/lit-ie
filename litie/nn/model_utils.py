from dataclasses import dataclass
from typing import Optional, List, Any, Tuple

import torch
from transformers import AlbertModel, AlbertPreTrainedModel, AlbertTokenizer
from transformers import BertModel, BertPreTrainedModel, BertTokenizerFast
from transformers import ErnieModel, ErniePreTrainedModel
from transformers import NezhaModel, NezhaPreTrainedModel
from transformers import XLNetModel, XLNetPreTrainedModel, XLNetTokenizer
from transformers.file_utils import ModelOutput

from .chinese_bert import ChineseBertModel, ChineseBertTokenizerFast
from .roformer import RoFormerModel, RoFormerPreTrainedModel

MODEL_MAP = {
    "bert": (BertModel, BertPreTrainedModel, "bert"),
    "ernie": (ErnieModel, ErniePreTrainedModel, "ernie"),
    "roformer": (RoFormerModel, RoFormerPreTrainedModel, "roformer"),
    "nezha": (NezhaModel, NezhaPreTrainedModel, "nezha"),
    "albert": (AlbertModel, AlbertPreTrainedModel, "albert"),
    "xlnet": (XLNetModel, XLNetPreTrainedModel, "transformer"),
    "chinese_bert": (ChineseBertModel, BertPreTrainedModel, "bert"),
}

TOKENIZER_MAP = {
    "bert": BertTokenizerFast,
    "ernie": BertTokenizerFast,
    "roformer": BertTokenizerFast,
    "nezha": BertTokenizerFast,
    "albert": AlbertTokenizer,
    "xlnet": XLNetTokenizer,
    "chinese_bert": ChineseBertTokenizerFast,
}


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
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

from .cnn import get_auto_cnn_ner_model, get_cnn_ner_model_config
from .crf import (
    get_auto_crf_ner_model,
    get_auto_cascade_crf_ner_model,
    get_auto_softmax_ner_model,
    get_crf_ner_model_config,
    get_cascade_crf_ner_model_config,
    get_softmax_ner_model_config,
)
from .global_pointer import get_auto_global_pointer_ner_model, get_global_pointer_ner_model_config
from .lear import get_auto_lear_ner_model, get_lear_ner_model_config
from .mrc import get_auto_mrc_ner_model, get_mrc_ner_model_config
from .span import get_auto_span_ner_model, get_span_ner_model_config
from .tplinker import get_auto_tplinker_ner_model, get_tplinker_ner_model_config
from .w2ner import get_auto_w2ner_ner_model, get_w2ner_model_config
from ...registry import BaseParent


class AutoNerTaskModelConfig(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, labels, **kwargs):
        return cls.registry[class_key](labels, **kwargs)


class AutoNerTaskModel(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, model_type="bert", **kwargs):
        return cls.registry[class_key](model_type, **kwargs)


AutoNerTaskModelConfig.add_to_registry("crf", get_crf_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("cascade_crf", get_cascade_crf_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("softmax", get_softmax_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("span", get_span_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("global_pointer", get_global_pointer_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("mrc", get_mrc_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("lear", get_lear_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("tplinker", get_tplinker_ner_model_config)
AutoNerTaskModelConfig.add_to_registry("w2ner", get_w2ner_model_config)
AutoNerTaskModelConfig.add_to_registry("cnn", get_cnn_ner_model_config)


AutoNerTaskModel.add_to_registry("crf", get_auto_crf_ner_model)
AutoNerTaskModel.add_to_registry("cascade_crf", get_auto_cascade_crf_ner_model)
AutoNerTaskModel.add_to_registry("softmax", get_auto_softmax_ner_model)
AutoNerTaskModel.add_to_registry("span", get_auto_span_ner_model)
AutoNerTaskModel.add_to_registry("global_pointer", get_auto_global_pointer_ner_model)
AutoNerTaskModel.add_to_registry("mrc", get_auto_mrc_ner_model)
AutoNerTaskModel.add_to_registry("lear", get_auto_lear_ner_model)
AutoNerTaskModel.add_to_registry("tplinker", get_auto_tplinker_ner_model)
AutoNerTaskModel.add_to_registry("w2ner", get_auto_w2ner_ner_model)
AutoNerTaskModel.add_to_registry("cnn", get_auto_cnn_ner_model)

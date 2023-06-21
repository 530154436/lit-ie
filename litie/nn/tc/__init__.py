from .fc import get_auto_fc_tc_model, get_fc_tc_model_config
from ...registry import BaseParent


class AutoTextClassificationModelConfig(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, labels, **kwargs):
        return cls.registry[class_key](labels, **kwargs)


class AutoTextClassificationTaskModel(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, model_type="bert", **kwargs):
        return cls.registry[class_key](model_type, **kwargs)


AutoTextClassificationModelConfig.add_to_registry("tc", get_fc_tc_model_config)

AutoTextClassificationTaskModel.add_to_registry("tc", get_auto_fc_tc_model)

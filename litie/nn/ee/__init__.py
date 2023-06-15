from .gplinker import get_auto_gplinker_ee_model, get_gplinker_ee_model_config
from ...registry import BaseParent


class AutoEventExtractionTaskModelConfig(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, labels, **kwargs):
        return cls.registry[class_key](labels, **kwargs)


class AutoEventExtractionTaskModel(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, model_type="bert", **kwargs):
        return cls.registry[class_key](model_type, **kwargs)


AutoEventExtractionTaskModelConfig.add_to_registry("gplinker", get_gplinker_ee_model_config)

AutoEventExtractionTaskModel.add_to_registry("gplinker", get_auto_gplinker_ee_model)

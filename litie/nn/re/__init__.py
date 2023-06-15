from .casrel import get_auto_casrel_re_model, get_casrel_re_model_config
from .gplinker import get_auto_gplinker_re_model, get_gplinker_re_model_config
from .grte import get_auto_grte_re_model, get_grte_re_model_config
from .onerel import get_auto_onerel_re_model, get_onerel_re_model_config
from .pfn import get_auto_pfn_re_model, get_pfn_re_model_config
from .prgc import get_auto_prgc_re_model, get_prgc_re_model_config
from .spn import get_auto_spn_re_model, get_spn_re_model_config
from .tplinker import get_auto_tplinker_re_model, get_tplinker_re_model_config
from ...registry import BaseParent


class AutoReTaskModelConfig(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, labels, **kwargs):
        return cls.registry[class_key](labels, **kwargs)


class AutoReTaskModel(BaseParent):

    registry = {}

    @classmethod
    def create(cls, class_key, model_type="bert", **kwargs):
        return cls.registry[class_key](model_type, **kwargs)


AutoReTaskModelConfig.add_to_registry("casrel", get_casrel_re_model_config)
AutoReTaskModelConfig.add_to_registry("gplinker", get_gplinker_re_model_config)
AutoReTaskModelConfig.add_to_registry("grte", get_grte_re_model_config)
AutoReTaskModelConfig.add_to_registry("pfn", get_pfn_re_model_config)
AutoReTaskModelConfig.add_to_registry("prgc", get_prgc_re_model_config)
AutoReTaskModelConfig.add_to_registry("spn", get_spn_re_model_config)
AutoReTaskModelConfig.add_to_registry("tplinker", get_tplinker_re_model_config)
AutoReTaskModelConfig.add_to_registry("onerel", get_onerel_re_model_config)

AutoReTaskModel.add_to_registry("casrel", get_auto_casrel_re_model)
AutoReTaskModel.add_to_registry("gplinker", get_auto_gplinker_re_model)
AutoReTaskModel.add_to_registry("grte", get_auto_grte_re_model)
AutoReTaskModel.add_to_registry("pfn", get_auto_pfn_re_model)
AutoReTaskModel.add_to_registry("prgc", get_auto_prgc_re_model)
AutoReTaskModel.add_to_registry("spn", get_auto_spn_re_model)
AutoReTaskModel.add_to_registry("tplinker", get_auto_tplinker_re_model)
AutoReTaskModel.add_to_registry("onerel", get_auto_onerel_re_model)

from .base import NerDataModule
from .cnn import CnnForNerDataModule
from .crf import CrfForNerDataModule, CascadeCrfForNerDataModule
from .global_pointer import GlobalPointerForNerDataModule
from .lear import LearForNerDataModule
from .mrc import MrcForNerDataModule
from .span import SpanForNerDataModule
from .tplinker import TPlinkerNerDataModule
from .w2ner import W2NerDataModule
from ...registry import BaseParent


class AutoNerDataModule(BaseParent):

    registry = {}


AutoNerDataModule.add_to_registry("softmax", CrfForNerDataModule)
AutoNerDataModule.add_to_registry(CrfForNerDataModule.config_name, CrfForNerDataModule)
AutoNerDataModule.add_to_registry(CascadeCrfForNerDataModule.config_name, CascadeCrfForNerDataModule)
AutoNerDataModule.add_to_registry(CnnForNerDataModule.config_name, CnnForNerDataModule)
AutoNerDataModule.add_to_registry(GlobalPointerForNerDataModule.config_name, GlobalPointerForNerDataModule)
AutoNerDataModule.add_to_registry(LearForNerDataModule.config_name, LearForNerDataModule)
AutoNerDataModule.add_to_registry(SpanForNerDataModule.config_name, SpanForNerDataModule)
AutoNerDataModule.add_to_registry(MrcForNerDataModule.config_name, MrcForNerDataModule)
AutoNerDataModule.add_to_registry(TPlinkerNerDataModule.config_name, TPlinkerNerDataModule)
AutoNerDataModule.add_to_registry(W2NerDataModule.config_name, W2NerDataModule)

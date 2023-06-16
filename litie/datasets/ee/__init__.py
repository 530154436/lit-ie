from .gplinker import GPLinkerForEeDataModule
from ...registry import BaseParent


class AutoEventExtractionDataModule(BaseParent):

    registry = {}


AutoEventExtractionDataModule.add_to_registry(GPLinkerForEeDataModule.config_name, GPLinkerForEeDataModule)

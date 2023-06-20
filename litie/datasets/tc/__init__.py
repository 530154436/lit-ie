from .base import TextClassificationDataModule
from ...registry import BaseParent


class AutoTextClassificationDataModule(BaseParent):

    registry = {}


AutoTextClassificationDataModule.add_to_registry("tc", TextClassificationDataModule)

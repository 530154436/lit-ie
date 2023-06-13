from .base import RelationExtractionDataModule
from .casrel import CasRelForReDataModule
from .gplinker import GPLinkerForReDataModule
from .grte import GRTEForReDataModule
from .pfn import PFNForReDataModule
from .prgc import PRGCForReDataModule
from .spn import SPNForReDataModule
from .tplinker import TPlinkerForREDataModule
from ...registry import BaseParent


class AutoReDataModule(BaseParent):

    registry = {}


AutoReDataModule.add_to_registry(CasRelForReDataModule.config_name, CasRelForReDataModule)
AutoReDataModule.add_to_registry(GPLinkerForReDataModule.config_name, GPLinkerForReDataModule)
AutoReDataModule.add_to_registry(GRTEForReDataModule.config_name, GRTEForReDataModule)
AutoReDataModule.add_to_registry(PFNForReDataModule.config_name, PFNForReDataModule)
AutoReDataModule.add_to_registry(PRGCForReDataModule.config_name, PRGCForReDataModule)
AutoReDataModule.add_to_registry(SPNForReDataModule.config_name, SPNForReDataModule)
AutoReDataModule.add_to_registry(TPlinkerForREDataModule.config_name, TPlinkerForREDataModule)

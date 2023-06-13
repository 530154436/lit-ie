import operator

from lightning_utilities.core.imports import compare_version, module_available

BOLTS_AVAILABLE = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.4.0")
BOLTS_GREATER_EQUAL_0_5_0 = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.5.0")
WANDB_AVAILABLE = module_available("wandb")
ACCELERATE_AVAILABLE = module_available("accelerate")
TORCH_SCATTER_AVAILABLE = module_available("torch_scatter")

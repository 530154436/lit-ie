import sys
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoConfig, PreTrainedTokenizerBase, get_scheduler, Adafactor, BertConfig
from transformers import PretrainedConfig, PreTrainedModel, Pipeline
from transformers import pipeline as hf_transformers_pipeline

from ..callbacks import AdversarialMethods
from ..utils.deepspeed import enable_transformers_pretrained_deepspeed_sharding
from ..utils.imports import ACCELERATE_AVAILABLE

if ACCELERATE_AVAILABLE:
    from accelerate import load_checkpoint_and_dispatch

from ..arguments import TrainingArguments
from ..utils.logger import logger


def verify_manual_optimization_support(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    if model.automatic_optimization:
        return
    trainer.gradient_clip_val = None
    trainer.accumulate_grad_batches = 1


class TaskEngine(pl.LightningModule):

    def __init__(
        self,
        model_type: str,
        task_model_name: str,
        pretrained_model_name_or_path: Optional[str] = None,
        config: Optional[PretrainedConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_config_kwargs: Optional[dict] = None,
        pipeline: Optional[Any] = None,
        pipeline_kwargs: Optional[dict] = None,
        load_weights: bool = True,
        deepspeed_sharding: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        base_model_class: Optional[PreTrainedModel] = None,
        parent_model_class: Optional[PreTrainedModel] = None,
        training_args: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.load_weights = load_weights

        self.model_type = model_type
        self.task_model_name = task_model_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._pipeline = pipeline

        self.model_config_kwargs = model_config_kwargs or {}
        self._pipeline_kwargs = pipeline_kwargs or {}
        self.training_args = training_args if training_args else TrainingArguments

        self.base_model_class = base_model_class
        self.parent_model_class = parent_model_class

        logger.debug(f"Training parameters: {self.training_args}")

        # optimizer kwargs
        self.deepspeed_sharding = deepspeed_sharding
        if not self.deepspeed_sharding:
            self.initialize_model(self.pretrained_model_name_or_path, config, model)

        # adversarial parameters
        if self.training_args.do_adv:
            self.adversarial = AdversarialMethods[self.training_args.adv_mode](
                model=self.model,
                emb_name=self.training_args.adv_embedding_name,
            )

            self.automatic_optimization = False
            k = "pytorch_lightning.trainer.configuration_validator"
            if k in sys.modules:
                setattr(sys.modules[k], '__verify_manual_optimization_support', verify_manual_optimization_support)

            self.training_step = self.adversarial_step

    def initialize_model(self, pretrained_model_name_or_path: str, config: PretrainedConfig, model: PreTrainedModel):
        """create and initialize the model to use with this task,
        Feel free to overwrite this method if you are initializing the model in a different way
        """
        if config is None:
            try:
                config, unused_kwargs = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    return_unused_kwargs=True,
                    **self.model_config_kwargs
                )
            except ValueError:
                config, unused_kwargs = BertConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    return_unused_kwargs=True,
                    **self.model_config_kwargs
                )

            for key, value in unused_kwargs.items():
                setattr(config, key, value)

        if model is None:
            model = self.get_auto_model(self.model_type, self.task_model_name)

        if self.load_weights:
            self.model = model.from_pretrained(
                pretrained_model_name_or_path, config=config
            )
        else:
            self.model = model.from_config(config)

    def configure_optimizers(self) -> Dict:
        optimizer = self.create_optimizer(self.model)
        scheduler = self.create_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer = None):
        """
        the learning rate scheduler.
        """
        warmup_steps = max(self.training_args.warmup_ratio, self.training_args.warmup_steps)
        if self.scheduler is None:
            num_training_steps, num_warmup_steps = self.compute_warmup(
                num_training_steps=-1,
                num_warmup_steps=warmup_steps,
            )

            self.scheduler = get_scheduler(
                str(self.training_args.lr_scheduler_type),
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        return self.scheduler

    def create_optimizer(self, model: torch.nn.Module):
        if self.optimizer is None:
            if bool(self.training_args.adafactor):
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = torch.optim.AdamW
                optimizer_kwargs = {
                    "eps": 1e-8,
                }

            optimizer_grouped_parameters = self.create_model_param_optimizer(model)
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                lr=float(self.training_args.learning_rate),
                **optimizer_kwargs,
            )

        return self.optimizer

    def create_model_param_optimizer(self, model: PreTrainedModel):
        """different learning rate for different modules
        """
        no_decay = ["bias", 'LayerNorm.weight']
        optimizer_grouped_parameters = []

        if float(self.training_args.other_learning_rate) != 0.0:
            base_model = getattr(model, model.base_model_prefix)
            base_model_param = list(base_model.named_parameters())

            base_model_param_ids = [id(p) for n, p in base_model_param]
            other_model_param = [(n, p) for n, p in model.named_parameters() if
                                 id(p) not in base_model_param_ids]

            optimizer_grouped_parameters.extend(
                self._param_optimizer(
                    base_model_param,
                    float(self.training_args.learning_rate),
                    no_decay,
                    float(self.training_args.weight_decay)
                )
            )
            optimizer_grouped_parameters.extend(
                self._param_optimizer(
                    other_model_param,
                    float(self.training_args.other_learning_rate),
                    no_decay,
                    float(self.training_args.weight_decay)
                )
            )
        else:
            all_model_param = list(model.named_parameters())
            optimizer_grouped_parameters.extend(
                self._param_optimizer(
                    all_model_param,
                    float(self.training_args.learning_rate),
                    no_decay,
                    float(self.training_args.weight_decay)
                )
            )

        return optimizer_grouped_parameters

    def _param_optimizer(self, params, learning_rate, no_decay, weight_decay):
        return [{"params": [p for n, p in params if all(nd not in n for nd in no_decay)], "weight_decay": weight_decay,
                 "lr": learning_rate},
                {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
                 "lr": learning_rate, }]

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: Optional[str] = None) -> None:
        self.configure_metrics(stage)
        if self.deepspeed_sharding and not hasattr(self, "model"):
            enable_transformers_pretrained_deepspeed_sharding(self)
            self.initialize_model(self.pretrained_model_name_or_path, self.config, self.model)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module, and initialize any data specific metrics.
        """
        pass

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "tokenizer")
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @property
    def pipeline_task(self) -> Optional[str]:
        """Override to define what HuggingFace pipeline task to use.
        Returns: Optional string to define what pipeline task to use.
        """
        return None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            if self.pipeline_task is None:
                raise RuntimeError("No task was defined for this model. Try overriding `hf_pipeline_task`")
            else:
                self._pipeline = hf_transformers_pipeline(
                    task=self.pipeline_task, model=self.model, tokenizer=self.tokenizer, **self._pipeline_kwargs
                )
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    def predict(self, *args, **kwargs) -> Any:
        return self.pipeline(*args, **kwargs)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        pipeline_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model: TaskEngine = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
        # update model with pipeline_kwargs override
        if pipeline_kwargs is not None:
            model._pipeline_kwargs.update(pipeline_kwargs)
        return model

    def load_checkpoint_and_dispatch(self, *args, **kwargs) -> None:
        """Use when loading checkpoint via accelerate for large model support.
        Useful for when loading sharded checkpoints.
        """
        self.model = load_checkpoint_and_dispatch(self.model, *args, **kwargs)

    @property
    def device_map(self) -> Dict:
        """
        Returns: Device Map as defined when using `load_checkpoint_and_dispatch`.
        """
        return self.model.hf_device_map

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        output_dir = Path(self.training_args.output_dir)
        save_path = output_dir.joinpath(f"{self.model_type}-{self.task_model_name}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def training_step(self, batch: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def adversarial_step(self, batch: Any, **kwargs: Any) -> torch.Tensor:
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        opt.zero_grad()

        loss = None
        if self.training_args.adv_mode == "fgm":
            loss = self.model(**batch)[0]
            self.manual_backward(loss)

            self.adversarial.attack(epsilon=self.training_args.adv_epsilon)
            loss = self.model(**batch)[0]
            self.manual_backward(loss)

        elif self.training_args.adv_mode == "pgd":
            loss = self.model(**batch)[0]
            self.manual_backward(loss)
            self.adversarial.backup_grad()
            for t in range(self.training_args.num_adv_attacks):
                self.adversarial.attack(
                    is_first_attack=(t == 0),
                    alpha=self.training_args.adv_alpha,
                    epsilon=self.training_args.adv_epsilon,
                )
                if t != self.training_args.num_adv_attacks - 1:
                    opt.zero_grad()
                else:
                    self.adversarial.restore_grad()
                loss = self.model(**batch)[0]
                self.manual_backward(loss)

        self.adversarial.restore()

        if self.training_args.max_grad_norm is not None:
            self.clip_gradients(opt, gradient_clip_val=self.training_args.max_grad_norm)

        opt.step()
        scheduler and scheduler.step()
        self.model.zero_grad()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def get_auto_model(self, model_type, task_model_name):
        return None

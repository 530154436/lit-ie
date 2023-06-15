from typing import Optional, Any, Union, List, Dict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import PreTrainedTokenizerBase, BertTokenizerFast, PreTrainedModel

from ..arguments import TrainingArguments, DataTrainingArguments, ModelArguments
from ..callbacks import LoggingCallback
from ..datasets.base import TaskDataModule
from ..engines import TaskEngine
from ..nn.model_utils import TOKENIZER_MAP
from ..utils.imports import WANDB_AVAILABLE
from ..utils.logger import logger


class BaseModel:

    config_name = "NLP"

    def __init__(
        self,
        task_model_name: Optional[str] = None,
        model_type: Optional[str] = "bert",
        model_name_or_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_config_kwargs: Optional[dict] = None,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[Any] = None,
        base_model_class: Optional[PreTrainedModel] = None,
        parent_model_class: Optional[PreTrainedModel] = None,
    ):
        self.task_model_name = task_model_name if task_model_name else model_args.task_name
        self.model_type = model_type if model_type else model_args.model_type
        self.model_name_or_path = model_name_or_path if model_name_or_path else model_args.model_name_or_path
        self.model_config_kwargs = model_config_kwargs
        self.tokenizer = tokenizer

        self.model_args = model_args
        self.training_args = training_args if training_args else TrainingArguments

        self.base_model_class = base_model_class
        self.parent_model_class = parent_model_class

    def create_engine(self) -> TaskEngine:
        raise NotImplementedError

    def create_data_module(
        self,
        data_args: DataTrainingArguments,
        is_chinese: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        labels: Optional[Union[Dict[str, int], List[Any]]] = None,
    ) -> TaskDataModule:
        raise NotImplementedError

    def finetune(
        self,
        data_args: DataTrainingArguments,
        is_chinese: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        labels: Optional[Union[Dict[str, Any], List[Any]]] = None,
        **trainer_kwargs,
    ):
        if self.tokenizer is None and self.model_name_or_path is not None:
            tokenizer_cls = TOKENIZER_MAP.get(self.model_type, BertTokenizerFast)
            self.tokenizer = tokenizer_cls.from_pretrained(self.model_name_or_path)

        self.data_module = self.create_data_module(data_args, is_chinese, cache_dir, labels=labels)
        self.engine = self.create_engine()
        trainer = self.make_trainer(**trainer_kwargs)

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {trainer.max_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {trainer.accumulate_grad_batches}")
        logger.info(f"  Total optimization steps = {trainer.estimated_stepping_batches:,}")

        trainer.fit(self.engine, self.data_module)

    def make_trainer(self, accelerator="gpu", devices=1, callbacks=None, **kwargs):
        max_epochs = int(self.training_args.num_train_epochs)
        val_check_interval = float(self.training_args.val_check_interval)
        gradient_clip_val = float(self.training_args.max_grad_norm)

        if "max_epochs" in kwargs:
            max_epochs = int(kwargs.pop("max_epochs"))
        if "val_check_interval" in kwargs:
            val_check_interval = float(kwargs.pop("val_check_interval"))
        if "gradient_clip_val" in kwargs:
            gradient_clip_val = float(kwargs.pop("gradient_clip_val"))

        if WANDB_AVAILABLE:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=self.config_name, name=f"{self.model_type}-{self.task_model_name}"
            )

        if callbacks is None:
            model_ckpt = ModelCheckpoint(
                dirpath=str(self.training_args.output_dir),
                filename="best_model",
                monitor="val_f1_micro",
                save_top_k=1,
                mode="max",
            )

            callbacks = [model_ckpt, LoggingCallback()]

        return Trainer(
            logger=wandb_logger if WANDB_AVAILABLE else None,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            val_check_interval=val_check_interval,
            gradient_clip_val=gradient_clip_val,
            callbacks=callbacks,
            **kwargs,
        )

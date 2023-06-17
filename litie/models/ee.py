from typing import Optional, Union, Any, List, Dict

from .base import BaseModel
from ..arguments import DataTrainingArguments
from ..datasets import AutoEventExtractionDataModule
from ..engines import EventExtractionEngine


class AutoEventExtractionModel(BaseModel):

    config_name = "Event Extraction"

    def create_engine(self):
        return EventExtractionEngine(
            self.model_type,
            self.task_model_name,
            pretrained_model_name_or_path=self.model_name_or_path,
            tokenizer=self.tokenizer,
            labels=self.data_module.schemas,
            model_config_kwargs=self.model_config_kwargs or {},
            training_args=self.training_args,
            base_model_class=self.base_model_class,
            parent_model_class=self.parent_model_class,
        )

    def create_data_module(
        self,
        data_args: DataTrainingArguments,
        is_chinese: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        labels: Optional[Union[Dict[str, int], List[Any]]] = None,
    ):
        return AutoEventExtractionDataModule.create(
            self.task_model_name,
            self.tokenizer,
            labels=labels,
            dataset_name=data_args.dataset_name,
            dataset_config_name=data_args.dataset_config_name,
            train_val_split=data_args.validation_split_percentage,
            train_file=data_args.train_file,
            validation_file=data_args.validation_file,
            train_batch_size=self.training_args.per_device_train_batch_size,
            validation_batch_size=self.training_args.per_device_eval_batch_size,
            num_workers=data_args.preprocessing_num_workers,
            train_max_length=data_args.train_max_length,
            validation_max_length=data_args.validation_max_length,
            limit_train_samples=data_args.max_train_samples,
            limit_val_samples=data_args.max_eval_samples,
            cache_dir=cache_dir if cache_dir else self.model_args.cache_dir,
            task_name=f"{self.model_type}-{self.task_model_name}",
            is_chinese=is_chinese if is_chinese else data_args.is_chinese,
        )

from typing import Optional, Union, Any, IO

from .base import BaseModel
from ..arguments import DataTrainingArguments
from ..datasets import AutoNerDataModule
from ..engines import AutoNerEngine
from ..pipelines import AutoNerPipeline
from ..registry import BaseParent


class NerModel(BaseModel):

    config_name = "Named Entity Recognition"

    def create_engine(self):
        return AutoNerEngine.create(
            self.task_model_name,
            self.model_type,
            pretrained_model_name_or_path=self.model_name_or_path,
            tokenizer=self.tokenizer,
            labels=self.data_module.schemas,
            model_config_kwargs=self.model_config_kwargs or {},
            training_args=self.training_args,
        )

    def create_data_module(
        self,
        data_args: DataTrainingArguments,
        is_chinese: Optional[bool] = False,
        cache_dir: Optional[str] = None,
    ):
        return AutoNerDataModule.create(
            self.task_model_name,
            self.tokenizer,
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


class AutoNerModel(BaseParent):

    @classmethod
    def create(cls, **kwargs):
        return NerModel(**kwargs)

    @classmethod
    def __getitem__(cls, key):
        return NerModel

    @classmethod
    def from_pretrained(
        cls,
        task_model_name: str,
        model_type: str,
        model_name_or_path: Union[str, IO],
        device: Optional[Any] = "cpu",
        max_seq_len: Optional[int] = 512,
        split_sentence: Optional[bool] = False,
        use_fp16: Optional[bool] = False,
        **kwargs,
    ):
        return AutoNerPipeline.create(
            task_model_name,
            model_type=model_type,
            model_name_or_path=model_name_or_path,
            device=device,
            use_fp16=use_fp16,
            max_seq_len=max_seq_len,
            split_sentence=split_sentence,
            **kwargs,
        )

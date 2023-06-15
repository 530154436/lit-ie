from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import torch

from .base import TaskEngine
from ..metrics import ExtractionScore
from ..nn.re import AutoReTaskModel, AutoReTaskModelConfig
from ..pipelines import RelationExtractionPipeline


class ReEngine(TaskEngine):
    def __init__(
        self,
        model_type: str,
        task_model_name: str,
        labels: Union[List[str], Dict[str, Any]],
        model_config_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        model_config_kwargs = model_config_kwargs or {}
        model_config_kwargs = self.get_auto_model_config(task_model_name, labels, **model_config_kwargs)

        super().__init__(model_type, task_model_name, model_config_kwargs=model_config_kwargs, **kwargs)
        self.labels = labels
        self.average = kwargs.get("average", "micro")

    def get_auto_model_config(self, task_model_name, labels, **kwargs):
        return AutoReTaskModelConfig.create(task_model_name, labels, **kwargs)

    def get_auto_model(self, model_type, task_model_name):
        return AutoReTaskModel.create(
            task_model_name,
            model_type,
            base_model=self.base_model_class,
            parent_model=self.parent_model_class,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, batch: Any) -> None:
        outputs = self.model(**batch)
        preds, labels = outputs["predictions"], outputs["groundtruths"]
        self.metric.update(labels, preds)

    def common_epoch_end(self, prefix: str):
        metric_dict = self.compute_metrics(mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        return metric_dict

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch)

    def on_validation_epoch_end(self):
        return self.common_epoch_end("val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch)

    def on_test_epoch_end(self):
        return self.common_epoch_end("test")

    def configure_metrics(self, _) -> None:
        self.metric = ExtractionScore(average=self.average)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def compute_metrics(self, mode="val") -> Dict[str, float]:
        p, r, f = self.metric.value()
        self.metric.reset()
        return {f"{mode}_precision": p, f"{mode}_recall": r, f"{mode}_f1_{self.average}": f}

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = RelationExtractionPipeline(
                self.task_model_name,
                model_type=self.model_type,
                model=self.model,
                tokenizer=self.tokenizer,
                load_weights=False,
                **self._pipeline_kwargs
            )
        return self._pipeline

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        batch = self._prepare_input(batch, device)
        if len(batch) == 0:
            raise ValueError(
                "The batch received was empty."
            )
        return batch

    def _prepare_input(self, data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=device)
            return data.to(**kwargs)
        return data

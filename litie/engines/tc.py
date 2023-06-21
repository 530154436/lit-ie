from typing import Any, Dict, List, Optional, Union

import torch
import torchmetrics

from .base import TaskEngine
from ..nn.tc import AutoTextClassificationTaskModel, AutoTextClassificationModelConfig
from ..pipelines import TextClassificationPipeline


class TextClassificationEngine(TaskEngine):
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
        self.num_classes = len(labels)

    def get_auto_model_config(self, task_model_name, labels, **kwargs):
        return AutoTextClassificationModelConfig.create(task_model_name, labels, **kwargs)

    def get_auto_model(self, model_type, task_model_name):
        return AutoTextClassificationTaskModel.create(
            task_model_name,
            model_type,
            base_model=self.base_model_class,
            parent_model=self.parent_model_class,
        )

    def common_step(self, batch: Any) -> None:
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        for k, metric in self.metrics.items():
            metric.update(preds, batch["labels"])

    def common_epoch_end(self, prefix: str):
        metric_dict = self.compute_metrics(mode=prefix)
        for k, metric in self.metrics.items():
            metric.reset()

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
        task = "multiclass"
        self.precision = torchmetrics.Precision(task, num_classes=self.num_classes, average="macro")
        self.recall = torchmetrics.Recall(task, num_classes=self.num_classes, average="macro")
        self.accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average="macro")

        self.metrics = {
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
        }

    def compute_metrics(self, mode="val") -> Dict[str, torch.Tensor]:
        return {f"{mode}_{k}": metric.compute() for k, metric in self.metrics.items()}

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = TextClassificationPipeline(
                self.task_model_name,
                model_type=self.model_type,
                model=self.model,
                tokenizer=self.tokenizer,
                load_weights=False,
                **self._pipeline_kwargs
            )
        return self._pipeline

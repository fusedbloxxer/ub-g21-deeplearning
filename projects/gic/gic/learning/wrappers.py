import typing as t
from typing import Any, TypedDict, Unpack, cast
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from torcheval.metrics import Mean, MulticlassF1Score
import lightning as tl
from abc import ABC, abstractmethod
import wandb.plot as pt
import wandb as wn


from .metrics import ConfusionMatrix


class ClassifierArgs(TypedDict):
    weight_decay: float
    num_classes: int
    lr: float


class ClassifierModule(ABC, tl.LightningModule):
    def __init__(self,
                 name: str,
                 lr: float,
                 num_classes: int,
                 weight_decay: float,
                 **kwargs: Any) -> None:
        super(ClassifierModule, self).__init__()

        # Common classification setup
        self.name = name
        self._lr = lr
        self._weight_decay = weight_decay
        self._num_classes = num_classes
        self._loss_fn = nn.CrossEntropyLoss()
        self.logger: WandbLogger

        # Common metrics analysis for classification
        self._metric_train_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self.device)
        self._metric_train_loss = Mean(device=self.device)
        self._metric_valid_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self.device)
        self._metric_valid_confm = ConfusionMatrix(device=self.device)
        self._metric_valid_loss = Mean(device=self.device)
        self.save_hyperparameters(kwargs)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def on_train_start(self) -> None:
        self._metric_train_f1_score.to(self.device)
        self._metric_train_loss.to(self.device)

    def on_train_epoch_start(self) -> None:
        self._metric_train_f1_score.reset()
        self._metric_train_loss.reset()

    def training_step(self, b: t.Tuple[Tensor, Tensor], b_idx: t.Any) -> Tensor:
        X, y_true = b
        logits: Tensor = self(X)
        loss: Tensor = self._loss_fn(logits, y_true)
        self._metric_train_loss.update(loss.detach())
        self._metric_train_f1_score.update(logits.detach(), y_true)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_f1_score', self._metric_train_f1_score.compute().item())
        self.log('train_loss', self._metric_train_loss.compute().item())

    def on_validation_start(self) -> None:
        self._metric_valid_f1_score.to(self.device)
        self._metric_valid_confm.to(self.device)
        self._metric_valid_loss.to(self.device)

    def on_validation_epoch_start(self) -> None:
        self._metric_valid_f1_score.reset()
        self._metric_valid_confm.reset()
        self._metric_valid_loss.reset()

    def validation_step(self, b: t.Tuple[Tensor, Tensor], b_idx: t.Any) -> Tensor:
        X, y_true = b
        logits: Tensor = self(X)
        loss: Tensor = self._loss_fn(logits, y_true)
        self._metric_valid_loss.update(loss)
        self._metric_valid_confm.update(logits, y_true)
        self._metric_valid_f1_score.update(logits, y_true)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('valid_f1_score', self._metric_valid_f1_score.compute().item())
        self.log('valid_loss', self._metric_valid_loss.compute().item())

    def test_step(self, b: Tensor, b_idx: t.Any) -> Tensor:
        return torch.argmax(self(b), dim=-1)

    def predict_step(self, b: Tensor, b_idx: t.Any) -> Tensor:
        return torch.argmax(self(b), dim=-1)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = om.AdamW(self.parameters(), betas=(0.9, 0.999), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = om.lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.65, 10, min_lr=5e-5, cooldown=5)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_f1_score',
            },
        }

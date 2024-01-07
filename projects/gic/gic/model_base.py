import typing as t
from typing import Any, TypedDict
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import WandbLogger as Logger
from lightning.pytorch import LightningDataModule
from lightning import Trainer
import lightning as tl
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from torcheval.metrics import Mean, MulticlassF1Score
from torcheval.metrics import Metric
from abc import ABC, abstractmethod
from optuna.trial import Trial as Run
import pathlib as pl
from functools import partial

from .setup import Config
from .train_metrics import ConfusionMatrix
from .data_dataloader import GICDataModule
from .data_transform import PreprocessTransform, RobustAugmentTransform as RATransform


############ Base Classification Module ############
class ClassifierArgs(TypedDict):
    lr: float
    num_classes: int
    weight_decay: float
    augment: bool
    augment_n: t.Optional[int]
    augment_m: t.Optional[int]


class ClassifierModule(ABC, tl.LightningModule):
    def __init__(self,
                 name: str,
                 lr: float,
                 augment: bool,
                 num_classes: int,
                 weight_decay: float,
                 augment_n: t.Optional[int],
                 augment_m: t.Optional[int],
                 **kwargs: Any) -> None:
        super(ClassifierModule, self).__init__()
        self.trainer: tl.Trainer
        self.logger: Logger

        # Perform preprocessing
        self.preprocess = PreprocessTransform()

        # Perform augmentations
        if augment:
            assert augment_n is not None
            assert augment_m is not None
            self.augment = RATransform(rand_n=augment_n, rand_m=augment_m)
        else:
            self.augment = nn.Identity()

        # Common classification setup
        self._lr= lr
        self.name = name
        self._num_classes= num_classes
        self._weight_decay= weight_decay
        self._loss_fn = nn.CrossEntropyLoss()

        # Common metrics analysis for classification
        self._metric_train_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self.device)
        self._metric_train_loss = Mean(device=self.device)
        self._metric_valid_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self.device)
        self._metric_valid_confm = ConfusionMatrix(device=self.device)
        self._metric_valid_loss = Mean(device=self.device)
        self.save_hyperparameters(ignore=['name'])

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def on_train_start(self) -> None:
        self._metric_train_f1_score.to(self.device)
        self._metric_train_loss.to(self.device)

    def on_train_epoch_start(self) -> None:
        self._metric_train_f1_score.reset()
        self._metric_train_loss.reset()

    def training_step(self, b: t.Tuple[Tensor, Tensor, Tensor], b_idx: t.Any) -> Tensor:
        X, y_blend, y_true = b
        logits: Tensor = self(X)
        loss: Tensor = self._loss_fn(logits, y_blend)
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
        optim = om.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = om.lr_scheduler.StepLR(optim, 20, 0.90, verbose=True)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
            },
        }

    def on_after_batch_transfer(self, batch: Tensor | t.List[Tensor], _: int) -> Any:
        if self.trainer.training:
            assert isinstance(batch, list)
            y_true = batch[1]
            batch = self.augment(batch[0], batch[1])
            batch = [self.preprocess(batch[0]), batch[1], y_true]
        elif self.trainer.validating or self.trainer.sanity_checking:
            assert isinstance(batch, list)
            batch = [self.preprocess(batch[0]), batch[1]]
        else:
            assert isinstance(batch, Tensor)
            batch = self.preprocess(batch)
        return batch


############ Base Objective ############
class ScoreObjective(ABC):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 epochs: int,
                 data_module: t.Callable[[], LightningDataModule],
                 logger_fn: partial[Logger]) -> None:
        super(ScoreObjective, self).__init__()

        # Data
        self._data_fn = data_module

        # Model info
        self._model_name = model_name
        self._batch_size = batch_size
        self._epochs = epochs

        # Logging setup
        self._logger: Logger
        self.__logger_fn = logger_fn
        self._name_format = r'{model_name}/optimize/{study_name}/{run_name}'

    def __call__(self, run: Run) -> t.Any:
        # Sample a model, train it over the data and log the progress
        self.__log_init(run)
        score: t.Any = self.search(run)
        self.__log_finish()
        return score

    @abstractmethod
    def search(self, run: Run) -> t.Any:
        # Sample a model, train it over the data and evaluate it at the end
        raise NotImplementedError()

    @abstractmethod
    def model(self, run: Run) -> t.Tuple[tl.LightningModule, Metric[Tensor]]:
        # Samples hiperparameters from the defined space and returns a model and a metric callback
        raise NotImplementedError()

    def __log_init(self, run: Run):
        # Initialize the logger for the current run
        run_name = self._name_format.format(model_name=self._model_name,
                                                 study_name=run.study.study_name,
                                                 run_name=run.number)
        self._logger = self.__logger_fn(name=run_name)

    def __log_finish(self) -> None:
        # End the current run before starting another one
        self._logger.experiment.finish()


class F1ScoreObjective(ScoreObjective):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 epochs: int,
                 data_module: t.Callable[[], LightningDataModule],
                 logger_fn: partial[Logger]) -> None:
        super(F1ScoreObjective, self).__init__(model_name, batch_size, epochs, data_module, logger_fn)

    def search(self, run: Run) -> float:
        # Create custom model using factory
        model, metric = self.model(run)

        # Prepare training setup
        trainer = Trainer(max_epochs=self._epochs, enable_checkpointing=False, logger=self._logger)

        # Keep track of the best hyperparams
        self._logger.log_hyperparams(params={
            **model.hparams,
            'epochs': self._epochs,
            'batch_size': self._batch_size
        })

        # Train the model
        trainer.fit(model, datamodule=self._data_fn())

        # Evaluate its performance and update the study
        return metric.compute().item()

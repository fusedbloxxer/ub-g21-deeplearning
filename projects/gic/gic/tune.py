import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch import optim as om
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torcheval.metrics import Mean, MulticlassF1Score
import typing as t
from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict, TypeVar, TypeAlias, Generic, Optional, Callable, Union, Tuple, Literal, overload
import tqdm as tm
import optuna as opt
from optuna import Trial
import optuna.integration.mlflow as opt_mlf
import pathlib as pl
from pathlib import Path
import wandb as wn

from . import wn_callback, CONST_NUM_CLASS
from .data import TrainSplit, load_batched_data
from .utils import forward_self
from .model import ResCNN


class HyperParameterSampler(object):
    def __init__(self, sampler: Callable[[Trial], Dict[str, t.Any]]) -> None:
        self.__sampler = sampler

    def __call__(self, trial: Trial) -> Dict[str, Any]:
        return self.__sampler(trial)


class Trainer(ABC):
    def __init__(self,
                 dataset_path: Path,
                 hps: HyperParameterSampler,
                 train_valid_split: TrainSplit = (0.8, 0.2),
                 seed: int = 7982,
                 num_workers=8,
                 prefetch_factor: Optional[int] = 4,
                 device=torch.device('cuda')
                 ) -> None:

        # Dataset
        self.num_classes_ = CONST_NUM_CLASS
        self.num_workers_ = num_workers
        self.dataset_path_ = dataset_path
        self.train_valid_split_ = train_valid_split
        self.prefetch_factor_ = prefetch_factor

        # Runtime
        self.gen_ = torch.Generator('cpu').manual_seed(seed)
        self.device_ = device

        # HyperParameters
        self._hps = hps

    @abstractmethod
    def __call__(self, trial: Trial) -> float:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def valid_step(self, epoch: int, trial: Trial):
        raise NotImplementedError()

    @overload
    def load_batched_data(self, batch_size: int, split: TrainSplit):
        ...
    @overload
    def load_batched_data(self, batch_size: int):
        ...
    def load_batched_data(self, batch_size: int, split: Optional[TrainSplit] = None):
        return load_batched_data(self.dataset_path_,
                                 t.cast(TrainSplit, split or self.train_valid_split_),
                                 seed=self.gen_.seed(),
                                 batch_size=batch_size,
                                 num_workers=self.num_workers_,
                                 prefetch_factor=self.prefetch_factor_)


class ClassificationTrainer(Trainer):
    def __init__(self,
                 model: type[nn.Module],
                 *args,
                 **kwargs) -> None:
        super(ClassificationTrainer, self).__init__(*args, **kwargs)
        self.model_factory = model
        self.loss_fn_ = nn.CrossEntropyLoss()

    @forward_self(wn_callback.track_in_wandb())
    def __call__(self, trial: Trial) -> float:
        # Sample hyperparameters from search space
        hparams = self._hps.__call__(trial)

        # Shuffle the datasets
        data_loaders = self.load_batched_data(hparams['batch_size'])
        self.train_lr_ = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[0])
        self.valid_lr_ = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[1])

        # Instantiate model from scratch to perform optimization
        self.model_ = self.model_factory(hparams).to(self.device_)

        # Sample an optimizer similar to Adam to optimize the model weights
        optim_type = t.cast(type[om.Adam], getattr(om, hparams['optimizer']))
        self.optim_ = optim_type(self.model_.parameters(),
                                 weight_decay=hparams['weight_decay'],
                                 lr=hparams['lr'])

        # Perform search by optimizing Validation F1-Score
        valid_f1_score: float = 0.0
        for epoch in tm.trange(hparams["epochs"], desc='epoch', position=0):
            train_f1_score = self.train_step(epoch)
            valid_f1_score = self.valid_step(epoch, trial)
        return valid_f1_score

    def train(self, hparams: Dict[str, Any]):
        # Initialize logging run, to ensure training goes well
        wn.init(**wn_callback._wandb_kwargs)

        # Merge validation into the training data and shuffle
        data_loaders = self.load_batched_data(hparams['batch_size'], 'train')
        self.train_lr_ = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[0])

        # Instantiate model from scratch to perform optimization
        self.model_ = self.model_factory(hparams).to(self.device_)

        # Use an optimizer similar to Adam to fit the model
        optim_type = t.cast(type[om.Adam], getattr(om, hparams['optimizer']))
        self.optim_ = optim_type(self.model_.parameters(),
                                 weight_decay=hparams['weight_decay'],
                                 lr=hparams['lr'])

        # Train the model
        train_f1_score = 0.0
        for epoch in tm.trange(hparams["epochs"], desc='epoch', position=0):
            train_f1_score = self.train_step(epoch)
        return train_f1_score

    def eval(self, hparams: Dict[str, Any]):
        # Merge validation into the training data and shuffle
        data_loaders = self.load_batched_data(hparams['batch_size'], 'disjoint')
        self.test_lr_ = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[2])

        # Ensure evaluation mode is active
        self.model_ = self.model_.eval().requires_grad_(False)

        # Evaluate the model across all testing data
        with tm.tqdm(desc='test_batch', total=len(self.test_lr_), position=1) as batch:
            # Keep the predictions in a single place across evaluation
            pred = torch.zeros((len(t.cast(t.Sequence, self.test_lr_.dataset)),), dtype=torch.int32, device=self.device_)

            # Perform inference
            for b, X in enumerate(self.test_lr_):
                X: Tensor = X.to(self.device_)
                y_pred = torch.argmax(self.model_(X), dim=-1)
                index_from = b * t.cast(int, self.test_lr_.batch_size)
                index_to = index_from + t.cast(int, self.test_lr_.batch_size)
                pred[index_from: index_to] = y_pred

        # Return list of predictions
        return pred.cpu()

    def train_step(self, epoch: int):
        # Ensure training mode is active
        self.model_ = self.model_.train(True).requires_grad_(True)
        metric_train_f1_score = MulticlassF1Score(num_classes=self.num_classes_, average='macro', device=self.device_)
        metric_train_loss = Mean(device=self.device_)

        # Perform a training iteration across the entire dataset
        with tm.tqdm(desc='train_batch', total=len(self.train_lr_), position=1) as batch:
            for X, y in self.train_lr_:
                # Send data to GPU
                X: Tensor = X.to(self.device_)
                y_true: Tensor = y.to(self.device_)
                # TODO: KORNIA

                # Train
                self.optim_.zero_grad()
                logits: Tensor = self.model_(X)
                loss: Tensor = self.loss_fn_(logits, y_true)
                loss.backward()
                self.optim_.step()

                # Track metrics
                metric_train_f1_score.update(logits.detach(), y_true)
                metric_train_loss.update(loss.detach())
                batch.update(1)

        # Log Metrics
        train_f1_score = metric_train_f1_score.compute().item()
        wn.log({'train_loss': metric_train_loss.compute().item(), 'epoch': epoch})
        wn.log({'train_f1_score': metric_train_f1_score.compute().item(), 'epoch': epoch})
        metric_train_f1_score.reset()
        metric_train_loss.reset()
        return train_f1_score

    def valid_step(self, epoch: int, trial: Trial):
        # Ensure evaluation mode is active
        self.model_ = self.model_.eval().requires_grad_(False)
        metric_valid_f1_score = MulticlassF1Score(
            num_classes=self.num_classes_, average='macro', device=self.device_)
        metric_valid_loss = Mean(device=self.device_)

        # Perform a validation iteration across the entire dataset
        with tm.tqdm(desc='valid_batch', total=len(self.valid_lr_), position=1) as batch:
            for X, y in self.valid_lr_:
                # Send data to GPU
                X: Tensor = X.to(self.device_)
                y_true: Tensor = y.to(self.device_)
                # TODO: KORNIA

                # Infer
                with torch.no_grad():
                    logits: Tensor = self.model_(X)

                # Track metrics
                loss: Tensor = self.loss_fn_(logits, y_true)
                metric_valid_f1_score.update(logits, y_true)
                metric_valid_loss.update(loss)
                batch.update(1)

        # Log Metrics
        valid_f1_score = metric_valid_f1_score.compute().item()
        wn.log({'valid_loss': metric_valid_loss.compute().item(), 'epoch': epoch})
        wn.log({'valid_f1_score': valid_f1_score, 'epoch': epoch})
        metric_valid_loss.reset()
        metric_valid_f1_score.reset()
        trial.report(valid_f1_score, epoch)

        # Cut the search path if it's underperforming
        if trial.should_prune():
            raise opt.TrialPruned()

        # Otherwise continue the search
        return valid_f1_score

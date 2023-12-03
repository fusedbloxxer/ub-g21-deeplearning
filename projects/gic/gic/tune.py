import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch import optim as om
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
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

from . import wn_callback, CONST_NUM_CLASS, LOG_PATH
from .data import GenImageAugment, TrainSplit, load_batched_data
from .profile import TrainProfiler
from .utils import forward_self


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
                 augment: bool=False,
                 seed: int = 7982,
                 num_workers=8,
                 profiling: bool=False,
                 mix_float: bool=False,
                 prefetch_factor: Optional[int] = 4,
                 device=torch.device('cuda')
                 ) -> None:
        # Dataset
        self._augment = augment
        self._num_classes = CONST_NUM_CLASS
        self._num_workers = num_workers
        self._dataset_path = dataset_path
        self._train_valid_split = train_valid_split
        self._prefetch_factor = prefetch_factor
        self._normalize_fn = GenImageAugment(normalize=True, augment=False)
        self._augment_fn = self._normalize_fn if not augment else GenImageAugment(normalize=True, augment=True)

        # Runtime
        self._gen = torch.Generator('cpu').manual_seed(seed)
        self._profiler = TrainProfiler(enable=profiling)
        self._mix_float = mix_float
        self._device = device

        # Metrics
        self._metric_train_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self._device)
        self._metric_valid_f1_score = MulticlassF1Score(num_classes=self._num_classes, average='macro', device=self._device)
        self._metric_train_loss = Mean(device=self._device)
        self._metric_valid_loss = Mean(device=self._device)

        # HyperParameters
        self._hps = hps

    @abstractmethod
    def __call__(self, trial: Trial) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _train_step(self, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def _valid_step(self, epoch: int, trial: Trial):
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def eval(self) -> nn.Module:
        raise NotImplementedError()

    @overload
    def load_batched_data(self, batch_size: int, split: TrainSplit):
        ...
    @overload
    def load_batched_data(self, batch_size: int):
        ...
    def load_batched_data(self, batch_size: int, split: Optional[TrainSplit] = None):
        return load_batched_data(self._dataset_path,
                                 t.cast(TrainSplit, split or self._train_valid_split),
                                 seed=self._gen.seed(),
                                 batch_size=batch_size,
                                 normalize=False,
                                 num_workers=self._num_workers,
                                 prefetch_factor=self._prefetch_factor)


class ClassificationTrainer(Trainer):
    def __init__(self,
                 model: type[nn.Module],
                 *args,
                 **kwargs) -> None:
        super(ClassificationTrainer, self).__init__(*args, **kwargs)
        self.__model_factory = model
        self.loss_fn = nn.CrossEntropyLoss()

    @forward_self(wn_callback.track_in_wandb())
    def __call__(self, trial: Trial) -> float:
        """Optimize hyperparams using Optuna by repeated sampling."""
        self.__hparams = self._hps.__call__(trial)
        self.__trial = trial
        wn.config = self.__hparams
        wn.config['augment'] = self._augment
        wn.log({ 'augment': self._augment })

        # Shuffle the datasets
        data_loaders = self.load_batched_data(self.__hparams['batch_size'])
        self.__train_lr = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[0])
        self.__valid_lr = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[1])

        # Instantiate model from scratch to perform optimization
        self.model = self.__model_factory(**self.__hparams).to(self._device)
        print(self.model)

        # Sample an optimizer similar to Adam to optimize the model weights
        optim_type = t.cast(type[om.Adam], getattr(om, self.__hparams['optimizer']))
        self.__optim = optim_type(self.model.parameters(),
                                  weight_decay=self.__hparams['weight_decay'],
                                  lr=self.__hparams['lr'])
        self.__optim_scheduler = om.lr_scheduler.ReduceLROnPlateau(self.__optim, 'max', 0.75, 10, min_lr=2e-4, cooldown=5, verbose=True)
        self.__grad_scaler = GradScaler()
        return self.__optimize()

    def __optimize(self) -> float:
        """Optimize hyperparams using Optuna by maximizing f1-score on validation subset."""
        for epoch in tm.trange(self.__hparams["epochs"], desc='epochs'):
            self._train_step(epoch)
            self._valid_step(epoch)
        return self._metric_valid_f1_score.compute().item()

    def _train_step(self, epoch: int):
        # Ensure training mode is active
        self.model.train(True)
        self._metric_train_loss.reset()
        self._metric_train_f1_score.reset()

        # Perform a training iteration across the entire dataset
        with tm.tqdm(desc='train_batch', total=len(self.__train_lr)) as batch:
            with self._profiler as profiler:
                for X, y in self.__train_lr:
                    # TODO: KORNIA
                    # Send data to GPU
                    X: Tensor = self._augment_fn(X.to(self._device))
                    y_true: Tensor = y.to(self._device)
                    self.__optim.zero_grad(set_to_none=True)

                    # Use MixedPrecision to leverage TensorCores (multiples of 8)
                    with autocast(self._device.type, torch.float16, enabled=self._mix_float):
                        logits: Tensor = self.model(X)
                        loss: Tensor = self.loss_fn(logits, y_true)

                    # Update grads
                    if self._mix_float:
                        t.cast(Tensor, self.__grad_scaler.scale(loss)).backward()
                        self.__grad_scaler.step(self.__optim)
                    else:
                        loss.backward()
                        self.__optim.step()

                    # Track metrics
                    self._metric_train_f1_score.update(logits.detach(), y_true)
                    self._metric_train_loss.update(loss.detach())

                    # Mark batch as done
                    if self._mix_float:
                        self.__grad_scaler.update()
                    profiler.step()
                    batch.update(1)
                self.__optim_scheduler.step(self._metric_train_f1_score.compute().item())

        # Log Metrics
        train_loss = self._metric_train_loss.compute().item()
        train_f1_score = self._metric_train_f1_score.compute().item()
        wn.log({'train_loss': train_loss, 'epoch': epoch})
        wn.log({'train_f1_score': train_f1_score, 'epoch': epoch})

    def _valid_step(self, epoch: int):
        # Ensure evaluation mode is active
        self.model = self.model.eval()
        self._metric_valid_f1_score.reset()
        self._metric_valid_loss.reset()

        # Perform a validation iteration across the entire dataset
        with tm.tqdm(desc='valid_batch', total=len(self.__valid_lr)) as batch:
            with torch.no_grad():
                for X, y in self.__valid_lr:
                    # TODO: KORNIA
                    # Send data to GPU
                    X: Tensor = self._normalize_fn(X.to(self._device))
                    y_true: Tensor = y.to(self._device)

                    # Infer
                    logits: Tensor = self.model(X)

                    # Track metrics
                    loss: Tensor = self.loss_fn(logits, y_true)
                    self._metric_valid_f1_score.update(logits, y_true)
                    self._metric_valid_loss.update(loss)
                    batch.update(1)

        # Log Metrics
        valid_loss = self._metric_valid_loss.compute().item()
        valid_f1_score = self._metric_valid_f1_score.compute().item()
        wn.log({'valid_loss': valid_loss, 'epoch': epoch})
        wn.log({'valid_f1_score': valid_f1_score, 'epoch': epoch})
        self.__trial.report(valid_f1_score, epoch)

        # # Prune if accuracy is not good enough
        # if self.__trial.should_prune():
        #     raise opt.TrialPruned()

    def train(self, hparams: Dict[str, Any]):
        # Initialize logging run, to ensure training goes well
        wn.init(**wn_callback._wandb_kwargs)
        wn.config['augment'] = self._augment
        wn.log({ 'augment': self._augment })

        # Merge validation into the training data and shuffle
        data_loaders = self.load_batched_data(hparams['batch_size'], 'train')
        self.__train_lr = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[0])

        # Instantiate model from scratch to perform optimization
        self.model = self.__model_factory(**hparams).to(self._device)

        # Use an optimizer similar to Adam to fit the model
        optim_type = t.cast(type[om.Adam], getattr(om, hparams['optimizer']))
        self.__optim = optim_type(self.model.parameters(),
                                  weight_decay=hparams['weight_decay'],
                                  lr=hparams['lr'])
        self.__optim_scheduler = om.lr_scheduler.ReduceLROnPlateau(self.__optim, 'max', 0.75, 10, min_lr=2e-4, cooldown=5, verbose=True)
        self.__grad_scaler = GradScaler()

        # Train the model
        for epoch in tm.trange(hparams["epochs"], desc='epoch', position=0):
            self._train_step(epoch)
        return self._metric_train_f1_score.compute().item()

    def eval(self, hparams: Dict[str, Any]):
        # Merge validation into the training data and shuffle
        data_loaders = self.load_batched_data(hparams['batch_size'], 'disjoint')
        self.test_lr_ = t.cast(DataLoader[Tuple[Tensor, Tensor]], data_loaders[2])

        # Ensure evaluation mode is active
        self.model = self.model.eval()

        # Evaluate the model across all testing data
        with tm.tqdm(desc='test_batch', total=len(self.test_lr_), position=1) as batch:
            # Keep the predictions in a single place across evaluation
            pred = torch.zeros((len(t.cast(t.Sequence, self.test_lr_.dataset)),), dtype=torch.int32, device=self._device)

            # Perform inference
            for b, X in enumerate(self.test_lr_):
                X: Tensor = self._normalize_fn(X.to(self._device))
                y_pred = torch.argmax(self.model(X), dim=-1)
                index_from = b * t.cast(int, self.test_lr_.batch_size)
                index_to = index_from + t.cast(int, self.test_lr_.batch_size)
                pred[index_from: index_to] = y_pred
                batch.update(1)

        # Return list of predictions
        return pred.cpu()

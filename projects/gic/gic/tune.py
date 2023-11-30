from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch import optim as om
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torcheval.metrics import Mean, MulticlassF1Score
import typing as t
from typing import TypedDict, TypeVar, TypeAlias, Generic, Optional, Callable, Union, Tuple
import tqdm as tm
import optuna as opt
from optuna import Trial
import optuna.integration.mlflow as opt_mlf
import pathlib as pl
from pathlib import Path
import wandb as wn

from .data import load_data
from .model import ResCNN
from . import wn_callback


class HyperParameterSpace(TypedDict):
    lr: float
    epochs: int
    optimizer: str
    batch_size: int
    weight_decay: float


class ResCNNHPSpace(HyperParameterSpace):
    pool: str
    dropout1d: float
    dropout2d: float
    conv_chan: int
    dens_chan: int
    activ_fn: str


HPS = TypeVar('HPS', bound=HyperParameterSpace)


class HyperParameterSampler(Generic[HPS]):
    def __init__(self, sampler: Callable[[Trial], HPS]) -> None:
        self.__sampler = sampler

    def __call__(self, trial: Trial) -> HPS:
        return self.__sampler(trial)


class Trainer(Generic[HPS]):
    def __init__(self,
                 num_classes: int,
                 hps: HyperParameterSampler[HPS],
                 dataset_path: Path,
                 train_valid_split: Tuple[float,float]|t.Literal['disjoint']=(0.8, 0.2),
                 device=torch.device('cuda'),
                 seed: int=7982,
                 prefetch_factor=4,
                 num_workers=8
                 ) -> None:
        super(Trainer, self).__init__()

        # Dataset
        self.num_classes_ = num_classes
        self.dataset_path_ = dataset_path
        self.num_workers_ = num_workers
        self.prefetch_factor_ = prefetch_factor
        self.train_valid_split_ = train_valid_split

        # Runtime
        self.gen_ = torch.Generator('cpu').manual_seed(seed)
        self.device_ = device

        # HyperParameters
        self.__hps = hps

    @wn_callback.track_in_wandb()
    def __call__(self, trial: Optional[opt.Trial], params: Optional[HPS]=None) -> float:
        hparams = self.__hps.__call__(trial) if params is None else params

        self.train_lr_, self.valid_lr_, _ = self.__create_loaders(hparams['batch_size'])

        self.model_ = ResCNN(**t.cast(t.Any, hparams)).to(self.device_)

        self.loss_fn_ = nn.CrossEntropyLoss()

        optim_type = t.cast(type[om.Adam], getattr(om, hparams['optimizer']))
        self.optim_ = optim_type(self.model_.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

        valid_f1_score: float = 0.0
        for epoch in tm.trange(hparams["epochs"], desc='epoch', position=0):
            self.train_step(epoch)

            if params is None:
                valid_f1_score = self.valid_step(epoch, trial)
        return valid_f1_score

    def train_step(self, epoch: int):
        self.model_ = self.model_.train(True).requires_grad_(True)
        metric_train_f1_score = MulticlassF1Score(num_classes=self.num_classes_, average='macro', device=self.device_)
        metric_train_loss = Mean(device=self.device_)

        with tm.tqdm(desc='train_batch', total=len(self.train_lr_), position=1) as batch:
            for X, y in self.train_lr_:
                # Send data to GPU
                # TODO: KORNIA
                X: Tensor = X.to(self.device_)
                y_true: Tensor = y.to(self.device_)

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

        wn.log({ 'train_loss': metric_train_loss.compute().item(), 'epoch': epoch })
        wn.log({ 'train_f1_score': metric_train_f1_score.compute().item(), 'epoch': epoch })
        metric_train_f1_score.reset()
        metric_train_loss.reset()

    def valid_step(self, epoch: int, trial: Trial):
        self.model_ = self.model_.eval().requires_grad_(False)
        metric_valid_f1_score = MulticlassF1Score(num_classes=self.num_classes_, average='macro', device=self.device_)
        metric_valid_loss = Mean(device=self.device_)

        with tm.tqdm(desc='valid_batch', total=len(self.valid_lr_), position=1) as batch:
            for X, y in self.valid_lr_:
                # Send data to GPU
                # TODO: KORNIA
                X: Tensor = X.to(self.device_)
                y_true: Tensor = y.to(self.device_)

                # Infer
                with torch.no_grad():
                    logits: Tensor = self.model_(X)

                # Track metrics
                loss: Tensor = self.loss_fn_(logits, y_true)
                metric_valid_f1_score.update(logits, y_true)
                metric_valid_loss.update(loss)
                batch.update(1)

        valid_f1_score = metric_valid_f1_score.compute().item()
        wn.log({ 'valid_loss': metric_valid_loss.compute().item(), 'epoch': epoch })
        wn.log({ 'valid_f1_score': valid_f1_score, 'epoch': epoch })
        metric_valid_loss.reset()
        metric_valid_f1_score.reset()
        trial.report(valid_f1_score, epoch)

        if trial.should_prune():
            raise opt.TrialPruned()

        return valid_f1_score

    def __create_loaders(self, batch_size: int):
        data = load_data(self.dataset_path_, True)
        train_ds = t.cast(Dataset[Tuple[Tensor, Tensor]], data[0])
        valid_ds = t.cast(Dataset[Tuple[Tensor, Tensor]], data[1])
        test_ds = t.cast(Dataset[Tensor], t.cast(t.Any, data)[2])

        if isinstance(self.train_valid_split_, tuple):
            train_ds = ConcatDataset([train_ds, valid_ds])
            train_ds, valid_ds = random_split(train_ds, self.train_valid_split_, self.gen_)

        train_dl = DataLoader(train_ds, batch_size, True, prefetch_factor=self.prefetch_factor_, num_workers=self.num_workers_, generator=self.gen_)
        valid_dl = DataLoader(valid_ds, batch_size, True, prefetch_factor=self.prefetch_factor_, num_workers=self.num_workers_, generator=self.gen_)
        test_dl = DataLoader(test_ds, batch_size, False, prefetch_factor=self.prefetch_factor_, num_workers=self.num_workers_, generator=self.gen_)
        return train_dl, valid_dl, test_dl

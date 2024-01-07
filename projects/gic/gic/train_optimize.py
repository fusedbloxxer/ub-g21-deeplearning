import sys as s
import torch
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt

from . import *
from .data_dataset import GICDataset
from .data_dataloader import GICDataModule
from .model_densecnn import DenseCNNObjective


# Find hyperparams that maximize the validation f1 score
sampler = opt.samplers.TPESampler(n_startup_trials=10)
s = opt.create_study(direction='maximize', sampler=sampler)
s.optimize(DenseCNNObjective(), n_trials=250, show_progress_bar=True)

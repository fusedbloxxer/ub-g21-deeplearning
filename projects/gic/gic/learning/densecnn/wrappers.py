import typing as t
from typing import Any, Unpack
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from torcheval.metrics import Mean, MulticlassF1Score
import lightning as tl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from .modules import DenseCNN, DenseCNNArgs
from ..wrappers import ClassifierModule, ClassifierArgs


class DenseCNNClassifierArgs(ClassifierArgs, DenseCNNArgs):
    pass


class DenseCNNClassifierModule(ClassifierModule):
    def __init__(self,
                 **kwargs: Unpack[DenseCNNClassifierArgs]):
        super(DenseCNNClassifierModule, self).__init__(name='DenseCNN', **kwargs)
        self.net_densecnn = DenseCNN(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net_densecnn(x)

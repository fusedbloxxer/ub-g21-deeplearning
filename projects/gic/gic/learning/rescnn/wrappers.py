import typing as t
from typing import Any, Unpack
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from torcheval.metrics import Mean, MulticlassF1Score
import lightning as tl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from .modules import ResCNN, ResCNNArgs
from ..wrappers import ClassifierModule, ClassifierArgs


class ResCNNClassifierArgs(ClassifierArgs, ResCNNArgs):
    pass


class ResCNNClassifierModule(ClassifierModule):
    def __init__(self,
                 **kwargs: Unpack[ResCNNClassifierArgs]):
        super(ResCNNClassifierModule, self).__init__(name='ResCNN', **kwargs)
        self.rescnn = ResCNN(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.rescnn(x)

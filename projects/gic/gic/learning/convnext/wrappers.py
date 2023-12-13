import typing as t
from typing import Any, Unpack
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from torcheval.metrics import Mean, MulticlassF1Score
import lightning as tl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from .modules import ConvNextNet, ConvNextArgs
from ..wrappers import ClassifierModule, ClassifierArgs


class ConvNextClassifierArgs(ClassifierArgs, ConvNextArgs):
    pass


class ConvNextClassifierModule(ClassifierModule):
    def __init__(self,
                 **kwargs: Unpack[ConvNextClassifierArgs]):
        super(ConvNextClassifierModule, self).__init__(name='ConvNext', **kwargs)
        self.convnextnet = ConvNextNet(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.convnextnet(x)
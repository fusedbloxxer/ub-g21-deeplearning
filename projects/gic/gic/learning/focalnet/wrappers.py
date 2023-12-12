import torch
from torch import Tensor
import typing as t
from typing import Any

from ..wrappers import ClassifierArgs, ClassifierModule
from .modules import FocalNetModule, FocalNetArgs


class FocalNetClassifierArgs(ClassifierArgs, FocalNetArgs):
    pass


class FocalNetClassifier(ClassifierModule):
    def __init__(self, **kwargs: t.Unpack[FocalNetClassifierArgs]) -> None:
        super(FocalNetClassifier, self).__init__(name='FocalNet', **kwargs)
        self.focalnet = FocalNetModule(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.focalnet(x)


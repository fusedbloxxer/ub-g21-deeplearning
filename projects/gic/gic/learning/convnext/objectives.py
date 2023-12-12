import typing as t
from typing import Any, cast
from lightning.pytorch.core import LightningModule
from optuna.trial import Trial as Run
from torch import Tensor
from torcheval.metrics import Metric

from ...data.dataset import GICDataset
from ..objectives import F1ScoreObjective
from .wrappers import ConvNextClassifierModule


class ConvNextObjective(F1ScoreObjective):
    def __init__(self) -> None:
        super(ConvNextObjective, self).__init__('ConvNext')

    def model(self, run: Run) -> t.Tuple[LightningModule, Metric[Tensor]]:
        model = ConvNextClassifierModule(
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 2e-4, 9e-4),
            weight_decay=run.suggest_float('weight_decay', 6e-6, 8e-2),
            patch=4,
            h=64,
            w=64,
            chan=run.suggest_int('chan', 32, 96, step=32),
            conv_dropout=run.suggest_float('conv_dropout', 0.2, 0.2),
            conv_layers=run.suggest_int('conv_layers', 1, 3, step=1),
            dense_dropout=run.suggest_float('dense_dropout', 0.1, 0.3),
            dense_features=run.suggest_int('dense_features', 128, 512, step=128),
            dense_layers=run.suggest_int('dense_layers', 0, 2, step=1),
            activ_fn=cast(t.Any, run.suggest_categorical('activ_fn', ['SiLU', 'GELU'])),
            patch_reduce=run.suggest_categorical('patch_reduce', [True, False]),
        )
        return model, model._metric_valid_f1_score

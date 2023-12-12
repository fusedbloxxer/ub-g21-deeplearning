import typing as t
from typing import Any, cast
from lightning.pytorch.core import LightningModule
from optuna.trial import Trial as Run
from torch import Tensor
from torcheval.metrics import Metric

from ...data.dataset import GICDataset
from ..objectives import F1ScoreObjective
from .wrappers import ResCNNClassifierModule


class ResCNNObjective(F1ScoreObjective):
    def __init__(self) -> None:
        super(ResCNNObjective, self).__init__('ResCNN')

    def model(self, run: Run) -> t.Tuple[LightningModule, Metric[Tensor]]:
        model = ResCNNClassifierModule(
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 2e-4, 9e-4),
            weight_decay=run.suggest_float('weight_decay', 6e-6, 8e-2),
            pool=cast(t.Any, run.suggest_categorical('pool', ['max', 'avg'])),
            dropout1d=run.suggest_float('dense_dropout', 0.2, 0.6),
            dropout2d=run.suggest_float('conv_dropout', 0.3, 0.6),
            conv_chan=run.suggest_int('conv_chan', 16, 32, step=8),
            dens_chan=run.suggest_int('dens_chan', 128, 512, 128),
            activ_fn=run.suggest_categorical('activ', ['ReLU', 'SiLU', 'GELU'])
        )
        return model, model._metric_valid_f1_score

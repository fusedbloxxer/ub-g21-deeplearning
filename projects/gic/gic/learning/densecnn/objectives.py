import typing as t
from typing import Any, cast
from lightning.pytorch.core import LightningModule
from optuna.trial import Trial as Run
from torch import Tensor
from torcheval.metrics import Metric

from ...data.dataset import GICDataset
from ..objectives import F1ScoreObjective
from .wrappers import DenseCNNClassifierModule


class DenseCNNObjective(F1ScoreObjective):
    def __init__(self) -> None:
        super(DenseCNNObjective, self).__init__('DenseCNN')

    def model(self, run: Run) -> t.Tuple[LightningModule, Metric[Tensor]]:
        model = DenseCNNClassifierModule(
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 2e-4, 9e-4),
            weight_decay=run.suggest_float('weight_decay', 6e-6, 8e-2),
            dense=run.suggest_int('dense', 128, 256, step=128),
            features=run.suggest_int('features', 12, 16, step=4),
            factor_c=run.suggest_float('factor_c', 1.0, 1.0, step=0.25),
            factor_t=run.suggest_float('factor_t', 1.0, 1.0, step=0.25),
            f_drop=run.suggest_float('f_drop', 0.3, 0.4),
            c_drop=run.suggest_float('c_drop', 0.2, 0.3),
            inner=run.suggest_int('inner', 3, 4, step=1),
            repeat=run.suggest_int('repeat', 3, 4, step=1),
            pool=cast(Any, run.suggest_categorical('pool', ['max'])),
            activ_fn=cast(Any, run.suggest_categorical('activ_fn', ['SiLU'])),
        )
        return model, model._metric_valid_f1_score

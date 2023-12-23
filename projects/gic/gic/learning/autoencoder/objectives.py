from lightning.pytorch.core import LightningModule
from typing import Any, Tuple, cast
from optuna.trial import Trial as Run
from torch import Tensor
from torcheval.metrics import Metric
import pathlib as pl

from ... import CKPT_PATH
from ...data.dataset import GICDataset
from ..objectives import F1ScoreObjective
from .wrappers import DAEClasifierModule


class DAEClasifierObjective(F1ScoreObjective):
    def __init__(self):
        super(DAEClasifierObjective, self).__init__('DAEClassifier')

    def model(self, run: Run) -> Tuple[LightningModule, Metric[Tensor]]:
        model = DAEClasifierModule(
            ckpt_path=CKPT_PATH,
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 5e-5, 2e-3),
            dropout=run.suggest_float('dropout', 0.25, 0.6),
            augment=True,
            augment_n=run.suggest_categorical('augment_n', [1, 2, 3]),
            augment_m=run.suggest_categorical('augment_m', [11, 5, 4, 9, 7]),
            weight_decay=run.suggest_float('weight_decay', 6e-4, 3e-2),
            activ_fn=run.suggest_categorical('activ_fn', ['silu', 'leak']),
            dense=run.suggest_categorical('dense', [64, 128, 256]),
            batch=cast(Any, run.suggest_categorical('batch', ['batch', None])),
        )
        return model, model._metric_valid_f1_score

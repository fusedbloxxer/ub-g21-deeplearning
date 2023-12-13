from lightning.pytorch.core import LightningModule
from typing import Any, Tuple
from optuna.trial import Trial as Run
from torch import Tensor
from torcheval.metrics import Metric

from ...data.dataset import GICDataset
from ..objectives import F1ScoreObjective
from .wrappers import FocalNetClassifier


class FocalNetObjective(F1ScoreObjective):
    def __init__(self):
        super(FocalNetObjective, self).__init__('FocalNet')

    def model(self, run: Run) -> Tuple[LightningModule, Metric[Tensor]]:
        model = FocalNetClassifier(
            groups=8,
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 2e-4, 9e-4),
            weight_decay=run.suggest_float('weight_decay', 6e-6, 8e-2),
            layers=run.suggest_int('layers', 1, 4, step=1),
            chan=run.suggest_int('chan', 32, 64, 8),
            activ_fn=run.suggest_categorical('activ_fn', ['SiLU', 'LeakyReLU']),
            norm_layer=run.suggest_categorical('norm_layer', ['batch']),
            repeat=run.suggest_int('repeat', 1, 4, step=1),
            dropout=run.suggest_float('dropout', 0.15, 0.5),
            drop_type=run.suggest_categorical('dropout_type', ['spatial']),
            reduce=run.suggest_categorical('reduce', ['max']),
            dense=run.suggest_categorical('dense', [224, 256, 278]),
            dropout_dense=run.suggest_float('dropout_dense', 0.35, 0.6),
            conv_order=run.suggest_categorical('conv_order', ["0 1 2", "0 2 1", "1 0 2", "1 2 0"])
        )
        return model, model._metric_valid_f1_score
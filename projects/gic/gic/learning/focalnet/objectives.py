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
            lr=4e-4,
            groups=8,
            norm_layer='batch',
            drop_type='spatial',
            num_classes=GICDataset.num_classes,
            augment=True,
            augment_n=1,
            augment_m=9,
            repeat=3,
            dropout=0.12,
            layers=2,
            chan=128,
            weight_decay=0.0016,
            dense=256,
            reduce="max",
            dropout_dense=0.30,
            activ_fn='LeakyReLU',
            conv_order="2 1 0"
        )
        return model, model._metric_valid_f1_score



            # lr=4e-4,
            # groups=8,
            # norm_layer='batch',
            # drop_type='spatial',
            # num_classes=GICDataset.num_classes,
            # augment=True,
            # augment_n=run.suggest_categorical('augment_n', [1, 2, 3]),
            # augment_m=run.suggest_categorical('augment_m', [11, 5, 4, 9, 7]),
            # repeat=run.suggest_int('repeat', 1, 4, step=1),
            # dropout=run.suggest_float('dropout', 0.1, 0.3),
            # layers=run.suggest_int('layers', 1, 3, step=1),
            # chan=run.suggest_categorical('chan', [32, 64, 96, 128]),
            # weight_decay=run.suggest_float('weight_decay', 1e-6, 6e-3),
            # dense=run.suggest_categorical('dense', [128, 256, 384]),
            # reduce=run.suggest_categorical('reduce', ['max', 'avg']),
            # dropout_dense=run.suggest_float('dropout_dense', 0.2, 0.5),
            # activ_fn=run.suggest_categorical('activ_fn', ['SiLU', 'LeakyReLU', 'GELU', 'ReLU']),
            # conv_order=run.suggest_categorical('conv_order', ["0 1 2", "2 1 0", "0 2 1", "1 0 2", "2 0 1", "1 2 0"])
import optuna as opt
import typing as t
from typing import Any, Tuple, Dict
from optuna.trial import Trial as Run

from ..wrappers import ClassifierArgs, ClassifierModule
from .wrappers import FocalNetClassifier
from .modules import FocalNetArgs


def factory(run: Run, cparams: ClassifierArgs) -> t.Tuple[t.Dict[str, Any], ClassifierModule]:
    hparams = FocalNetArgs(
        groups=8,
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
    return t.cast(t.Dict[str, t.Any], hparams), FocalNetClassifier(**hparams, **cparams)

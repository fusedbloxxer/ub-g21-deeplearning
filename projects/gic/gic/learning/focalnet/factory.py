import optuna as opt
import typing as t
from typing import Any, Tuple, Dict
from optuna.trial import Trial as Run

from ..wrappers import ClassifierArgs, ClassifierModule
from .wrappers import FocalNetClassifier
from .modules import FocalNetArgs


def factory(run: Run, cparams: ClassifierArgs) -> t.Tuple[t.Dict[str, Any], ClassifierModule]:
    hparams = FocalNetArgs(
        chan=run.suggest_categorical('chan', [16, 32]),
        activ_fn=run.suggest_categorical('activ_fn', ['SiLU', 'GELU', 'LeakyReLU', 'ReLU']),
        norm_layer=run.suggest_categorical('norm_layer', ['layer', 'group', 'batch']),
        repeat=run.suggest_int('repeat', 1, 5, step=1),
        groups=run.suggest_categorical('groups', [2, 4, 8, 16, 32]),
        dropout=run.suggest_float('dropout', 0.1, 0.4),
        drop_type=run.suggest_categorical('dropout_type', ['channel', 'spatial']),
        reduce=run.suggest_categorical('reduce', ['max', 'avg', 'conv']),
        dense=run.suggest_categorical('dense', [128, 256, 512]),
        dropout_dense=run.suggest_float('dropout_dense', 0.2, 0.5),
        conv_order=run.suggest_categorical('conv_order', ["0 1 2", "0 2 1", "1 0 2", "1 2 0"])
    )
    return t.cast(t.Dict[str, t.Any], hparams), FocalNetClassifier(**hparams, **cparams)

import typing as t
from typing import Any, Unpack, cast
from optuna.trial import Trial as Run

from ..wrappers import ClassifierArgs, ClassifierModule
from .wrappers import DenseCNNClassifierModule
from .modules import DenseCNNArgs


def factory(run: Run, cparams: ClassifierArgs) -> t.Tuple[t.Dict[str, Any], ClassifierModule]:
    # Sample Model HyperParameters
    hparams = DenseCNNArgs(
        dense=run.suggest_int('dense', 128, 1024, step=128),
        features=run.suggest_int('features', 8, 32, step=4),
        factor_c=run.suggest_float('factor_c', 0.25, 1.0, step=0.25),
        factor_t=run.suggest_float('factor_t', 0.25, 1.0, step=0.25),
        f_drop=run.suggest_float('f_drop', 0.3, 0.4),
        c_drop=run.suggest_float('c_drop', 0.2, 0.3),
        inner=run.suggest_int('inner', 2, 5, step=1),
        repeat=run.suggest_int('repeat', 2, 5, step=1),
        pool=cast(Any, run.suggest_categorical('pool', ['max', 'avg'])),
        activ_fn=cast(Any, run.suggest_categorical('activ_fn', ['SiLU', 'GELU', 'LeakyReLU'])),
    )

    # Construct custom model for the current run
    return t.cast(t.Dict[str, Any], hparams), DenseCNNClassifierModule(**hparams, **cparams)

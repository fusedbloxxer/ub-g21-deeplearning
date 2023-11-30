import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna.study.study import ObjectiveFuncType


with try_import() as _imports:
    import wandb


@experimental_class("2.9.0")
class WeightsAndBiasesCallback:
    """Callback to track Optuna trials with Weights & Biases.

    This callback enables tracking of Optuna study in
    Weights & Biases. The study is tracked as a single experiment
    run, where all suggested hyperparameters and optimized metrics
    are logged and plotted as a function of optimizer steps.

    .. note::
        User needs to be logged in to Weights & Biases before
        using this callback in online mode. For more information, please
        refer to `wandb setup <https://docs.wandb.ai/quickstart#1-set-up-wandb>`_.

    .. note::
        Users who want to run multiple Optuna studies within the same process
        should call ``wandb.finish()`` between subsequent calls to
        ``study.optimize()``. Calling ``wandb.finish()`` is not necessary
        if you are running one Optuna study per process.

    .. note::
        To ensure correct trial order in Weights & Biases, this callback
        should only be used with ``study.optimize(n_jobs=1)``.


    Example:

        Add Weights & Biases callback to Optuna optimization.

        .. code::

            import optuna
            from optuna.integration.wandb import WeightsAndBiasesCallback


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()

            wandb_kwargs = {"project": "my-project"}
            wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

            study.optimize(objective, n_trials=10, callbacks=[wandbc])



        Weights & Biases logging in multirun mode.

        .. code::

            import optuna
            from optuna.integration.wandb import WeightsAndBiasesCallback

            wandb_kwargs = {"project": "my-project"}
            wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)


            @wandbc.track_in_wandb()
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()
            study.optimize(objective, n_trials=10, callbacks=[wandbc])


    Args:
        metric_name:
            Name assigned to optimized metric. In case of multi-objective optimization,
            list of names can be passed. Those names will be assigned
            to metrics in the order returned by objective function.
            If single name is provided, or this argument is left to default value,
            it will be broadcasted to each objective with a number suffix in order
            returned by objective function e.g. two objectives and default metric name
            will be logged as ``value_0`` and ``value_1``. The number of metrics must be
            the same as the number of values objective function returns.
        wandb_kwargs:
            Set of arguments passed when initializing Weights & Biases run.
            Please refer to `Weights & Biases API documentation
            <https://docs.wandb.ai/ref/python/init>`_ for more details.
        as_multirun:
            Creates new runs for each trial. Useful for generating W&B Sweeps like
            panels (for ex., parameter importance, parallel coordinates, etc).

    """

    def __init__(
        self,
        metric_name: Union[str, Sequence[str]] = "value",
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        as_multirun: bool = False,
    ) -> None:

        _imports.check()

        if not isinstance(metric_name, Sequence):
            raise TypeError(
                "Expected metric_name to be string or sequence of strings, got {}.".format(
                    type(metric_name)
                )
            )

        self._metric_name = metric_name
        self._wandb_kwargs = wandb_kwargs or {}
        self._as_multirun = as_multirun

        if not self._as_multirun:
            self._initialize_run()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        if isinstance(self._metric_name, str):
            if len(trial.values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = ["{}_{}".format(self._metric_name, i)
                         for i in range(len(trial.values))]

            else:
                names = [self._metric_name]

        else:
            if len(self._metric_name) != len(trial.values):
                raise ValueError(
                    "Running multi-objective optimization "
                    "with {} objective values, but {} names specified. "
                    "Match objective values and names, or use default broadcasting.".format(
                        len(trial.values), len(self._metric_name)
                    )
                )

            else:
                names = [*self._metric_name]

        metrics = {name: value for name, value in zip(names, trial.values)}

        if self._as_multirun:
            metrics["trial_number"] = trial.number

        attributes = {"direction": [d.name for d in study.directions]}

        step = trial.number if wandb.run else None
        run = wandb.run

        # Might create extra runs if a user logs in wandb but doesn't use the decorator.

        if not run:
            run = self._initialize_run()
            run.name = f"trial/{trial.number}/{run.name}"

        run.log({**trial.params, **metrics}, step=step)

        if self._as_multirun:
            run.config.update({**attributes, **trial.params})
            run.tags = tuple(self._wandb_kwargs.get(
                "tags", ())) + (study.study_name,)
            run.finish()
        else:
            run.config.update(attributes)

    @experimental_func("3.0.0")
    def track_in_wandb(self) -> Callable:
        """Decorator for using W&B for logging inside the objective function.

        The run is initialized with the same ``wandb_kwargs`` that are passed to the callback.
        All the metrics from inside the objective function will be logged into the same run
        which stores the parameters for a given trial.

        Example:

            Add additional logging to Weights & Biases.

            .. code::

                import optuna
                from optuna.integration.wandb import WeightsAndBiasesCallback
                import wandb

                wandb_kwargs = {"project": "my-project"}
                wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)


                @wandbc.track_in_wandb()
                def objective(trial):
                    x = trial.suggest_float("x", -10, 10)
                    wandb.log({"power": 2, "base of metric": x - 2})

                    return (x - 2) ** 2


                study = optuna.create_study()
                study.optimize(objective, n_trials=10, callbacks=[wandbc])


        Returns:
            Objective function with W&B tracking enabled.
        """

        def decorator(func: ObjectiveFuncType) -> ObjectiveFuncType:
            @functools.wraps(func)
            def wrapper(
                *args: Tuple[Any], **kwargs: Dict[Any, Any]
            ) -> Union[float, Sequence[float]]:
                trial = self._search_trial_instance(*args, **kwargs)

                # Uses global run when `as_multirun` is set to False.
                run = wandb.run
                if not run:
                    run = self._initialize_run()
                    run.name = f"trial/{trial.number}/{run.name}"

                return func(*args, **kwargs)  # type: ignore[arg-type]

            return wrapper  # type: ignore[return-value]

        return decorator

    def _initialize_run(self) -> Any:
        """Initializes Weights & Biases run."""
        run = wandb.init(**self._wandb_kwargs)
        return run

    @staticmethod
    def _search_trial_instance(*args: Tuple[Any], **kwargs: Dict[Any, Any]) -> optuna.trial.BaseTrial:
        for arg in args:
            if isinstance(arg, optuna.trial.BaseTrial):
                return arg

        for value in kwargs.values():
            if isinstance(value, optuna.trial.BaseTrial):
                return value

        assert False, "Should not reach here."


import torch
import os as so
import numpy as ny
import random as rng
import pathlib as pl
import optuna as opt
import typing as t
import wandb as wn
from git import Repo

# Development
debug = True
release = False

# Paths
ROOT_PATH = pl.Path('..')
GIT_PATH = ROOT_PATH / '..' / '..'
DATA_PATH = ROOT_PATH / 'data'
CKPT_PATH = ROOT_PATH / 'ckpt'
SUBMISSIONS_PATH = ROOT_PATH / 'submissions'
NOTEBOOKS_PATH = ROOT_PATH / 'notebooks'

# Versioning
git_repo = Repo(GIT_PATH)
SUBMISSION_PATH = SUBMISSIONS_PATH / f'{git_repo.active_branch.name}.csv'

# Tracking
so.environ['WANDB_NOTEBOOK_NAME'] = str(NOTEBOOKS_PATH / 'main.ipynb')
login_settings = {}
init_settings = {
    'project': 'Generated Image Classification',
    'tags': ['test', git_repo.active_branch.name]
}

# Use an account only during development
if not release:
    so.environ['WANDB_MODE'] = 'online'
    login_settings.update({
        'anonymous': 'never',
    })
    init_settings.update({})
else:
    so.environ['WANDB_MODE'] = 'offline'
    login_settings.update({
        'anonymous': 'must',
    })
    init_settings.update({
        'mode': 'disabled',
    })

# Init Tracking
wn.login(**t.cast(t.Any, login_settings))
wn.init(**t.cast(t.Any, init_settings))

# Integrate Tracking with Optuna
wn_callback = WeightsAndBiasesCallback(
    wandb_kwargs=init_settings,
    metric_name='f1_score',
    as_multirun=True,
)

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefetch_factor = 4
num_workers = 8
batch_size = 32

# Reproducibility
SEED = 7982
rng.seed(SEED)
ny.random.seed(SEED)
torch.manual_seed(SEED)
gen_numpy = ny.random.default_rng(SEED)
gen_torch = torch.Generator('cpu').manual_seed(SEED)

# Constants
CONST_NUM_CLASS = 100

# Miscellaneous
db_uri = r'sqlite:///test.db'
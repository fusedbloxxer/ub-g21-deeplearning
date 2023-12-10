import typing as t
from typing import Any, Callable, Dict, Tuple, cast
import optuna as opt
from optuna.trial import Trial as Run
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb as wn

from .. import DATA_PATH, PROJECT_NAME, LOG_PATH
from ..utils import PTLWrapper
from ..data.dataset import GICDataset
from ..data.dataloader import GICDataLoader
from .wrappers import ClassifierArgs, ClassifierModule


def search(run: Run, factory: Callable[[Run, ClassifierArgs], Tuple[Dict[str, Any], ClassifierModule]]) -> float:
    # Sample Common Params
    cparams = ClassifierArgs(
        num_classes=GICDataset.num_classes,
        lr=run.suggest_float('lr', 7e-4, 1e-3),
        weight_decay=run.suggest_float('weight_decay', 4e-6, 8e-3),
    )

    # Create custom model using factory
    hparams, dnn = factory(run, cparams)

    # Sample Training Settings
    batch_size = 32
    epochs = run.suggest_int('epochs', 100, 100, step=25)
    augment = run.suggest_categorical('augment', [False, True])
    logger = WandbLogger(project=PROJECT_NAME, name=f'{dnn.name}/opt/{run.study.study_name}/{run.number}', save_dir=LOG_PATH)

    # Prepare training setup
    loader = GICDataLoader(DATA_PATH, batch_size, augment)
    pruner = PTLWrapper(run, monitor="valid_f1_score")
    trainer = Trainer(max_epochs=epochs, enable_checkpointing=False, num_sanity_val_steps=0, logger=logger, callbacks=[pruner])
    trainer.logger.log_hyperparams(params={ **hparams, **cparams, 'augment': augment, 'batch_size': batch_size, 'epochs': epochs })

    # Perform training
    trainer.fit(dnn, datamodule=loader); wn.finish()
    return dnn._metric_valid_f1_score.compute().item()

import typing as t
import optuna as opt
from optuna.trial import Trial as Run
import lightning as tl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torcheval.metrics import Metric
import wandb as wn
from abc import ABC, abstractmethod
import random as rng

from ..data.dataset import GICDataset
from ..data.dataloader import GICDataModule
from .. import DATA_PATH, PROJECT_NAME, LOG_PATH, IS_RELEASE, SUBMISSION_NAME
from .wrappers import ClassifierArgs, ClassifierModule


class ScoreObjective(ABC):
    def __init__(self, model_name: str, **kwargs) -> None:
        super(ScoreObjective, self).__init__()
        self._logger: WandbLogger
        self._model_name = model_name
        self._name_format = r'{model_name}/optimize/{study_name}/{run_name}'

    def __call__(self, run: Run) -> t.Any:
        self.__log_init(run)
        score: t.Any = self.search(run)
        self.__log_finish()
        return score

    @abstractmethod
    def search(self, run: Run) -> t.Any:
        raise NotImplementedError()

    @abstractmethod
    def model(self, run: Run) -> t.Tuple[tl.LightningModule, Metric[Tensor]]:
        raise NotImplementedError()

    def __log_init(self, run: Run):
        run_name = self._name_format.format(model_name=self._model_name,
                                                  study_name=run.study.study_name,
                                                  run_name=run.number)
        self._logger = WandbLogger(dir=LOG_PATH,
                                   offline=IS_RELEASE,
                                   anonymous=IS_RELEASE,
                                   project=PROJECT_NAME,
                                   name=run_name)

    def __log_finish(self) -> None:
        self._logger.experiment.finish()


class F1ScoreObjective(ScoreObjective):
    def __init__(self, model_name: str, **kwargs) -> None:
        super(F1ScoreObjective, self).__init__(model_name, **kwargs)

    def search(self, run: Run) -> float:
        # Create custom model using factory
        model, metric = self.model(run)

        # Sample Training Settings
        batch_size: int = 32
        epochs: int = 60

        # Prepare training setup
        loader = GICDataModule(DATA_PATH, batch_size)
        trainer = Trainer(max_epochs=epochs, enable_checkpointing=False, logger=self._logger)

        # Keep track of the best hyperparams
        self._logger.log_hyperparams(params={ **model.hparams, 'epochs': epochs, 'batch_size': batch_size })

        # Perform training
        trainer.fit(model, datamodule=loader)
        return metric.compute().item()

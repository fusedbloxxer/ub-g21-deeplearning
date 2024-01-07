import typing as t
from typing import Any, Dict, Generic, TypeVar, Type, Tuple, cast
from functools import partial
import torch
from torch import Tensor, inference_mode
from torcheval.metrics.functional import multiclass_f1_score as f1_score
from torch.utils.data import DataLoader
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt
from lightning import LightningDataModule as DataModule
from lightning.pytorch.loggers import WandbLogger as Logger
from datetime import datetime as dt
import wandb as wn
import pathlib as pl

from .model_base import ClassifierModule, ClassifierArgs
from .data_dataset import GICDataset


Model = TypeVar('Model', bound=ClassifierModule, covariant=True)
Params = TypeVar('Params', bound=ClassifierArgs)


class BaggingEnsemble(Generic[Model, Params]):
    def __init__(self,
                 model_type: Type[Model],
                 n_models: int,
                 args: Params,
                 ckpt_path: pl.Path,
                 log_path: pl.Path,
                 device: torch.device,
                 project_name: str,
                 logger_fn: partial[Logger]):
        super(BaggingEnsemble, self).__init__()
        assert n_models > 1, 'at least two models may be specified'

        # Create generic bagging ensemble members
        self.models = [model_type(**cast(Dict[str, Any], args)) for _ in range(n_models)]
        self.model_name = self.models[-1].name
        self.model_type = model_type
        self.ckpt_save_dir = ckpt_path / 'ensemble' / self.model_name
        self.device = device
        self.log_path = log_path
        self.logger_fn = logger_fn
        self.project_name = project_name

    def fit(self, epochs: int, data: DataModule, validate: bool=False) -> None:
        start_time = dt.now().strftime(r'%d_%b_%Y_%H:%M')
        for i in range(len(self)):
            run_logger = self.logger_fn(
                name=f"Ensemble_{self.model_name}_{i + 1}_{len(self)}_Train_{start_time}"
            )
            model_ckpt = ModelCkpt(
                save_top_k=-1,
                every_n_epochs=1,
                dirpath=self.ckpt_save_dir,
                save_on_train_epoch_end=True,
                filename=f"{i + 1}_{len(self)}_{{epoch:03d}}",
            )
            trainer = Trainer(
                check_val_every_n_epoch=1 if validate else 0,
                limit_val_batches=1.0 if validate else 0,
                enable_checkpointing=True,
                num_sanity_val_steps=0,
                callbacks=[model_ckpt],
                logger=run_logger,
                max_epochs=epochs,
            )
            trainer.fit(self.models[i], data)
            wn.finish()

    def load_ensemble(self, epoch: int, version: int | None=None) -> None:
        v = f'-v{version}' if version else ''
        for i in range(1, len(self) + 1):
            self.models[i - 1] = self.model_type.load_from_checkpoint(self.ckpt_save_dir / f"{i}_{len(self)}_epoch={epoch:03d}{v}.ckpt")

    def validate(self, epochs: int, data: DataModule, version: int | None=None) -> None:
        start_time: str = dt.now().strftime(r'%d_%b_%Y_%H:%M')
        data.setup('validate')
        valid_dl: DataLoader[Tuple[Tensor, Tensor]] = data.val_dataloader()
        y_true: Tensor = torch.cat(list(map(lambda x: x[1], iter(valid_dl))), dim=0)
        wn.init(
            name=f"Ensemble_{self.model_name}_Valid_{start_time}",
            project=self.project_name,
            dir=self.log_path,
        )
        for epoch in range(epochs):
            self.load_ensemble(epoch, version)
            y_pred = self.predict(data, 'valid', setup=False).sum(dim=1).argmax(dim=-1)
            score = f1_score(y_pred, y_true, num_classes=GICDataset.num_classes, average='macro').item()
            wn.log({ 'valid_f1_score': score }, step=epoch)
            wn.log({ 'epoch': epoch }, step=epoch)
        wn.finish()

    def predict(self, data: DataModule, mode: t.Literal['valid', 'test'], setup: bool=True) -> Tensor:
        # Prepare data
        if setup:
            data.setup('validate' if mode == 'valid' else 'predict')
        pred_dl: DataLoader[Tensor] = data.predict_dataloader() if mode == 'test' else data.val_dataloader()

        # Evaluation mode
        for i in range(len(self)):
            self.models[i] = self.models[i].eval()

        # Peform inference
        y_pred = []
        for batch in pred_dl:
            if   mode == 'valid':
                X: Tensor = batch[0].to(self.device)
            elif mode == 'test':
                X: Tensor = batch.to(self.device)
            else:
                raise Exception(f'invalid predict mode: {mode}')
            X = self.models[0].preprocess(X)

            # Obtain outputs from members
            e_pred = []
            for i in range(len(self)):
                self.models[i] = self.models[i].to(self.device)
                with inference_mode(True):
                    e_pred.append(self.models[i].forward(X).cpu())
                self.models[i] = self.models[i].to('cpu')

            # Aggregate outputs from members
            y_pred.append(torch.stack(e_pred, dim=1))

        # Gather all outputs in the batch dim
        return torch.cat(y_pred, dim=0)

    def __len__(self) -> int:
        return len(self.models)

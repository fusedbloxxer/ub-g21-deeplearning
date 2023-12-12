import typing as t
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch.optim as om
from torch import Tensor
import lightning as tl
from torcheval.metrics import Mean
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ...data.transform import tr_perturb, tr_preprocess
from .modules import DenoisingAutoEncoder


class DAEModule(tl.LightningModule):
    def __init__(self,
                 chan: int,
                 lowdim: int,
                 layers: int,
                 drop_input: float,
                 drop_inside: float,
                 **kwargs) -> None:
        super(DAEModule, self).__init__()
        self.name = 'DAE'

        # Create model to be able to denoise & reconstruct images
        self.autoencoder = DenoisingAutoEncoder(3, chan, 3, lowdim, 64, 64, layers, drop_input, drop_inside)
        self._loss = nn.L1Loss()

        # Reconstruction metrics
        self._metric_train_loss = Mean(device=self.device)
        self._metric_valid_loss = Mean(device=self.device)

        # Analytics
        self.logger: WandbLogger
        self.save_hyperparameters({
            'chan': chan,
            'layers': layers,
            'lowdim': lowdim,
            'drop_input': drop_input,
            'drop_inside': drop_inside,
        })

    def forward(self, x: Tensor) -> Tensor:
        return self.autoencoder(x)

    def on_train_start(self) -> None:
        self._metric_train_loss.to(self.device)

    def on_train_epoch_start(self) -> None:
        self._metric_train_loss.reset()

    def training_step(self, b: t.Tuple[Tensor, Tensor], b_idx: t.Any) -> Tensor:
        # Retrieve noisy image
        X_true, _ = b
        X_noisy: Tensor = X_true

        # Denoise image
        X_pred: Tensor = self(X_noisy)

        # Compute reconstruction weighted loss
        loss: Tensor = self._loss(X_pred, X_true)

        # Track loss across epoch
        self._metric_train_loss.update(loss.detach())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('dae_train_loss', self._metric_train_loss.compute().item())

    def on_validation_start(self) -> None:
        self._metric_valid_loss.to(self.device)

    def on_validation_epoch_start(self) -> None:
        self._metric_valid_loss.reset()

    def validation_step(self, b: t.Tuple[Tensor, Tensor], b_idx: t.Any) -> Tensor:
        # Retrieve noisy image
        X_true, _ = b
        X_noisy: Tensor = X_true

        # Denoise image
        X_pred: Tensor = self(X_noisy)

        # Compute reconstruction weighted loss
        loss: Tensor = self._loss(X_pred, X_true)

        # Track loss across epoch
        self._metric_valid_loss.update(loss.detach())
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('dae_valid_loss', self._metric_valid_loss.compute().item())

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = om.Adam(self.parameters(), betas=(0.9, 0.999), lr=1e-3)
        scheduler = om.lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.75, 4, min_lr=8e-5, cooldown=5, threshold=7e-3)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'dae_train_loss',
            },
        }

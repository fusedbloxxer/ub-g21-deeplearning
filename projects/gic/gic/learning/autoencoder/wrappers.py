import typing as t
from typing import Any
import torch.nn as nn
import torch.optim as om
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch import Tensor
import lightning as tl
from torcheval.metrics import Mean
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from ...data.transform import MaskingNoiseTransform, AugmentTransform, PreprocessTransform
from ..wrappers import ClassifierArgs, ClassifierModule
from .modules import ConvAutoEncoder, DAEClassifierArgs, DAEClassifier


class DAEModule(tl.LightningModule):
    def __init__(self, chan: int, latent: int) -> None:
        super(DAEModule, self).__init__()
        self.save_hyperparameters()
        self.logger: WandbLogger
        self.name: str = 'DAE'

        # Transformations
        self.masking_noise = MaskingNoiseTransform()
        self.preprocess = PreprocessTransform()
        self.augment = AugmentTransform()

        # Create models
        self.autoencoder = ConvAutoEncoder(chan, latent)
        self._loss = nn.MSELoss(reduction='mean')

        # Reconstruction metrics
        self._metric_train_loss = Mean(device=self.device)
        self._metric_valid_loss = Mean(device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.autoencoder.forward(x)

    def on_train_start(self) -> None:
        self._metric_train_loss.to(self.device)

    def on_train_epoch_start(self) -> None:
        self._metric_train_loss.reset()

    def training_step(self, batch: Tensor, _: t.Any) -> Tensor:
        # Retrieve and augment data
        X_true = batch
        X_noisy: Tensor = self.masking_noise(batch)

        # Perform denoising
        X_denoised: Tensor = self.autoencoder.forward(X_noisy)

        # Compute denoising & reconstruction losses
        loss_denoised: Tensor = self._loss(X_denoised, X_true)

        # Track loss across epoch
        self._metric_train_loss.update(loss_denoised.detach())
        return loss_denoised

    def on_train_epoch_end(self) -> None:
        self.log('dae_train_loss', self._metric_train_loss.compute().item())

    def on_validation_start(self) -> None:
        self._metric_valid_loss.to(self.device)

    def on_validation_epoch_start(self) -> None:
        self._metric_valid_loss.reset()

    def validation_step(self, batch: Tensor, _: t.Any) -> Tensor:
        # Retrieve noisy image
        X_true = batch

        # Denoise image
        X_pred: Tensor = self(X_true)

        # Compute reconstruction weighted loss
        loss: Tensor = self._loss(X_pred, X_true)

        # Track loss across epoch
        self._metric_valid_loss.update(loss.detach())
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('dae_valid_loss', self._metric_valid_loss.compute().item())

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = om.RMSprop(self.parameters(), lr=1e-4)
        sch = CosineAnnealingWarmRestarts(opt, 15, 2, 1e-6, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'frequency': 1,
                'scheduler': sch,
                'interval': 'epoch',
                'monitor': 'dae_valid_loss',
            }
        }

    def on_after_batch_transfer(self, batch: Tensor, _: int) -> Tensor:
        if self.trainer.training:
            return self.augment(batch[0])
        elif self.trainer.validating or self.trainer.sanity_checking:
            return batch[0]
        else:
            return batch


class DAEClasifierModuleArgs(ClassifierArgs, DAEClassifierArgs):
    pass


class DAEClasifierModule(ClassifierModule):
    def __init__(self, **kwargs: t.Unpack[DAEClasifierModuleArgs]) -> None:
        super(DAEClasifierModule, self).__init__(name='DAEClassifier', **kwargs)
        self.dae_classifier = DAEClassifier(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.dae_classifier(x)

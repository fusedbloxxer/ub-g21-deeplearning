import typing as t
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import pathlib as pl
from collections import OrderedDict
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningDataModule
import torch.optim as om
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torcheval.metrics import Mean
import lightning as tl
from optuna.trial import Trial as Run
from lightning.pytorch.loggers import WandbLogger as Logger
from lightning.pytorch import LightningModule
from torch import Tensor
from torcheval.metrics import Metric
from functools import partial

from .model_base import F1ScoreObjective
from .model_base import ClassifierArgs, ClassifierModule
from .data_transform import PreprocessTransform, MaskingNoiseTransform, AugmentTransform
from .data_dataset import GICDataset


############ Modules ############
class ConvBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 ochan: int,
                 h: int,
                 w: int,
                 idim: t.Literal['reduce', 'expand'] | None = None,
                 odim: t.Literal['reduce', 'expand'] | None = None,
                 out_activ_fn: bool = True) -> None:
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()

        # Input Layers
        match idim:
            case None:
                pass
            case 'reduce':
                self.layers.append(nn.MaxPool2d(2, 2))
            case 'expand':
                self.layers.append(nn.UpsamplingNearest2d(scale_factor=2))
                self.layers.append(
                    nn.Conv2d(ichan, ichan, 3, 1, padding='same'))
            case _:
                raise ValueError(
                    'input dimension {} is not supported'.format(idim))

        # Hidden Layers
        self.layers.append(nn.Conv2d(ichan, ichan, 3, 1, padding='same'))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv2d(ichan, ochan, 3, 1, padding='same'))
        self.layers.append(nn.LeakyReLU() if out_activ_fn else nn.Identity())

        # Output Layers
        match odim:
            case None:
                pass
            case 'reduce':
                self.layers.append(nn.MaxPool2d(2, 2))
            case 'expand':
                self.layers.append(nn.UpsamplingNearest2d(scale_factor=2))
                self.layers.append(
                    nn.Conv2d(ochan, ochan, 3, 1, padding='same'))
            case _:
                raise ValueError(
                    'output dimension {} is not supported'.format(idim))

        # Skip Layer
        dim = 0
        dim += 1 if idim == 'expand' else -1 if idim == 'reduce' else 0
        dim += 1 if odim == 'expand' else -1 if odim == 'reduce' else 0
        self.skip = nn.Sequential()

        # Adjust spatial dimension
        if dim > 0:
            self.skip.append(nn.UpsamplingNearest2d(scale_factor=2 * dim))
        elif dim < 0:
            self.skip.append(nn.MaxPool2d((2, 2), -2 * dim))

        # Adjust channels
        self.skip.append(nn.Conv2d(ichan, ochan, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + self.skip(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self, chan: int, latent: int) -> None:
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, chan * 1, 64, 64, odim='reduce'),
            ConvBlock(chan * 1, chan * 2, 32, 32, odim='reduce'),
            ConvBlock(chan * 2, chan * 4, 16, 16, odim='reduce'),
            ConvBlock(chan * 4,   latent,  8,  8, odim='reduce'),
        )

        self.decoder = nn.Sequential(
            ConvBlock(latent, chan * 4,  8,  8, idim='expand'),
            ConvBlock(chan * 4, chan * 2, 16, 16, idim='expand'),
            ConvBlock(chan * 2, chan * 1, 32, 32, idim='expand'),
            ConvBlock(chan * 1,        3, 64, 64,
                      idim='expand', out_activ_fn=False),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


############ Neural Network ############
class DAEClassifierArgs(t.TypedDict):
    ckpt_path: pl.Path
    dense: int
    activ_fn: str
    dropout: float | None
    batch: str | None


class DAEClassifier(nn.Module):
    def __init__(self,
                 ckpt_path: pl.Path,
                 dense: int,
                 activ_fn: str,
                 dropout: float | None = None,
                 batch: str | None = None,
                 **kwargs) -> None:
        super(DAEClassifier, self).__init__()

        # Use pretrained denoising autoencoder
        self.autoencoder = ConvAutoEncoder(64, 64)
        self.autoencoder.load_state_dict(OrderedDict(list(map(lambda x: (x[0].replace('autoencoder.', ''), x[1]), torch.load(str(ckpt_path / 'dae.pt')).items()))))

        # Choose an activation
        activ = nn.ModuleDict({'silu': nn.SiLU(), 'leak': nn.LeakyReLU()})[activ_fn]

        # Train a classifier on top
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(64, dense),
            nn.BatchNorm1d(dense) if batch else nn.Identity(),
            activ,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(dense, dense),
            nn.BatchNorm1d(dense) if batch else nn.Identity(),
            activ,
            nn.Linear(dense, 100),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.autoencoder.encode(x)
        x = self.classifier.forward(x)
        return x


############ Classification Module ############
class DAEClasifierArgs(ClassifierArgs, DAEClassifierArgs):
    pass


class DAEClasifier(ClassifierModule):
    def __init__(self, **kwargs: t.Unpack[DAEClasifierArgs]) -> None:
        super(DAEClasifier, self).__init__(name=t.cast(t.Any, kwargs.pop(t.cast(t.Any, 'name'), 'DAEClassifier')), **kwargs)
        self.dae_classifier = DAEClassifier(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.dae_classifier(x)


############ Denoising Module ############
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
        opt = om.Adam(self.parameters(), lr=1e-4)
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


############ Objective ############
class DAEClasifierObjective(F1ScoreObjective):
    def __init__(self,
                 batch_size: int,
                 epochs: int,
                 data_module: t.Callable[[], LightningDataModule],
                 logger_fn: partial[Logger],
                 ckpt_path: pl.Path):
        super(DAEClasifierObjective, self).__init__('DAEClassifier', batch_size, epochs, data_module, logger_fn)
        self.ckpt_path = ckpt_path

    def model(self, run: Run) -> Tuple[LightningModule, Metric[Tensor]]:
        model = DAEClasifier(
            lr=6e-4,
            batch='batch',
            ckpt_path=self.ckpt_path,
            num_classes=GICDataset.num_classes,
            dropout=run.suggest_float('dropout', 0.25, 0.40),
            augment=True,
            augment_n=run.suggest_categorical('augment_n', [1, 2, 3]),
            augment_m=run.suggest_categorical('augment_m', [11, 5, 4, 9, 7]),
            weight_decay=run.suggest_float('weight_decay', 6e-4, 3e-3),
            activ_fn=run.suggest_categorical('activ_fn', ['silu', 'leak']),
            dense=run.suggest_categorical('dense', [224, 256]),
        )
        return model, model._metric_valid_f1_score

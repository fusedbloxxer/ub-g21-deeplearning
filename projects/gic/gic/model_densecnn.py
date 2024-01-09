import typing as t
from typing import Unpack
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as om
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger as Logger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from optuna.trial import Trial as Run
from torcheval.metrics import Metric
from functools import partial

from .data_dataset import GICDataset
from .model_base import F1ScoreObjective
from .model_base import ClassifierArgs, ClassifierModule


############ Modules ############
def create_activ_fn(activ_fn: t.Literal['SiLU', 'GELU', 'ReLU', 'LeakyReLU']):
    """Creates an activation function."""
    match activ_fn:
        case      'SiLU': return nn.SiLU()
        case      'GELU': return nn.GELU()
        case      'ReLU': return nn.ReLU()
        case 'LeakyReLU': return nn.LeakyReLU()
        case _: raise ValueError('activation function {} is not supported'.format(activ_fn))


def create_pool(reduce: t.Literal['avg', 'max']):
    """Creates an activation function."""
    match reduce:
        case 'avg': return nn.AvgPool2d(3, 2, 1)
        case 'max': return nn.MaxPool2d(3, 2, 1)
        case _: raise ValueError('reduce operation {} is not supported'.format(reduce))


class DenseConvBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 features: int,
                 kernel: int,
                 pad: int,
                 activ_fn: t.Literal['SiLU', 'GELU', 'ReLU', 'LeakyReLU'],
                 drop=0.3) -> None:
        super(DenseConvBlock, self).__init__()
        self.bn_layer = nn.BatchNorm2d(ichan)
        self.activ_fn = create_activ_fn(activ_fn)
        self.conv_layer = nn.Conv2d(ichan, features, kernel, 1, pad, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn_layer(x)
        x = self.activ_fn(x)
        x = self.drop(x)
        x = self.conv_layer(x)
        return x


class DenseCompConvBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 features: int,
                 factor_c: float,
                 activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU'],
                 drop=0.3) -> None:
        super(DenseCompConvBlock, self).__init__()
        assert 0 <= factor_c <= 1, 'invalid factor_c, must be in [0, 1]'

        # Dynamically sized block
        self.layers = nn.Sequential()

        # Add bottleneck
        if factor_c < 1:
            self.layers.add_module('comp_block', DenseConvBlock(
                ichan, int(ichan * factor_c), 1, 0, activ_fn, drop))

        # Add convolution
        self.layers.add_module('conv_block', DenseConvBlock(
            int(ichan * factor_c), features, 3, 1, activ_fn, drop))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 features: int,
                 factor_c: float,
                 activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU'],
                 repeat: int,
                 drop=0.3) -> None:
        super(DenseBlock, self).__init__()

        # Dynamically sized block
        self.layers = nn.ModuleList()

        # Add multiple inner blocks
        for l in range(repeat):
            self.layers.append(DenseCompConvBlock(
                ichan + l * features, features, factor_c, activ_fn, drop))

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = torch.cat([x, l(x)], dim=1)
        return x


class DenseDownBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 factor_t: float,
                 activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU'],
                 pool: t.Literal['max', 'avg'],
                 drop=0.3) -> None:
        super(DenseDownBlock, self).__init__()

        # Reduce the number of input channels
        self.down_conv = DenseConvBlock(ichan, int(
            ichan * factor_t), 1, 0, activ_fn, drop)

        # Reduce the spatial dimension
        self.down_size = create_pool(pool)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_size(self.down_conv(x))


############ Neural Network ############
class DenseCNNArgs(t.TypedDict):
    dense: int
    features: int
    factor_c: float
    factor_t: float
    activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU']
    pool: t.Literal['max', 'avg']
    repeat: int
    inner: int
    f_drop: float
    c_drop: float


class DenseCNN(nn.Module):
    def __init__(self,
                 dense: int = 128,
                 features: int = 24,
                 factor_c: float = 1.0,
                 factor_t: float = 1.0,
                 activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU'] = 'SiLU',
                 pool: t.Literal['max', 'avg'] = 'max',
                 repeat: int = 3,
                 inner: int = 3,
                 f_drop=0.2,
                 c_drop=0.1,
                 **kwargs) -> None:
        super(DenseCNN, self).__init__()

        # Allow initial layer to extract multiple features
        chan = int(3 * features * 3)
        self.input_conv = nn.Conv2d(3, chan, 1, 1, 0)

        # Chain multiple blocks
        self.dense_layers = nn.Sequential()
        for _ in range(repeat):
            self.dense_layers.append(DenseBlock(
                chan, features, factor_c, activ_fn, inner, c_drop))
            chan += features * inner
            self.dense_layers.append(DenseDownBlock(
                chan, factor_t, activ_fn, pool, c_drop))
            chan = int(chan * factor_t)
        self.dense_layers.append(nn.AdaptiveMaxPool2d(1))
        self.dense_layers.append(nn.Flatten())

        # Apply classification
        self.classifier = nn.Sequential(
            nn.Linear(in_features=chan, out_features=dense),
            nn.BatchNorm1d(dense),
            create_activ_fn(activ_fn),

            nn.Dropout(f_drop),
            nn.Linear(in_features=dense, out_features=dense),
            nn.BatchNorm1d(dense),
            create_activ_fn(activ_fn),

            nn.Linear(in_features=dense, out_features=100)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Extract local information from the image
        x = self.input_conv(x)
        x = self.dense_layers(x)
        x = self.classifier(x)
        return x


############ Classification Module ############
class DenseCNNClassifierArgs(ClassifierArgs, DenseCNNArgs):
    pass


class DenseCNNClassifier(ClassifierModule):
    def __init__(self, **kwargs: Unpack[DenseCNNClassifierArgs]):
        super(DenseCNNClassifier, self).__init__(name=t.cast(t.Any, kwargs.pop(t.cast(t.Any, 'name'), 'DenseCNN')), **kwargs)
        self.net_densecnn = DenseCNN(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.net_densecnn(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = om.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = om.lr_scheduler.StepLR(optim, 10, 0.85, verbose=True)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
            },
        }


############ Objective ############
class DenseCNNObjective(F1ScoreObjective):
    def __init__(self,
                 batch_size: int,
                 epochs: int,
                 data_module: t.Callable[[], LightningDataModule],
                 logger_fn: partial[Logger]) -> None:
        super(DenseCNNObjective, self).__init__('DenseCNN', batch_size, epochs, data_module, logger_fn)

    def model(self, run: Run) -> t.Tuple[LightningModule, Metric[Tensor]]:
        model = DenseCNNClassifier(
            num_classes=GICDataset.num_classes,
            lr=run.suggest_float('lr', 9e-5, 6e-4),
            inner=run.suggest_categorical('inner', [1, 2, 3, 4]),
            repeat=run.suggest_categorical('repeat', [1, 2, 3, 4]),
            features=run.suggest_categorical('features', [8, 16, 24, 32]),
            augment=run.suggest_categorical('augment', [True]),
            augment_n=run.suggest_categorical('augment_n', [1, 2, 3]),
            augment_m=run.suggest_categorical('augment_m', [11, 5, 4, 9, 7]),
            f_drop=run.suggest_float('f_drop', 0.2, 0.30),
            c_drop=run.suggest_float('c_drop', 0.1, 0.15),
            dense=run.suggest_categorical('dense', [224, 256, 312]),
            weight_decay=run.suggest_float('weight_decay', 1e-4, 6e-3),
            factor_c=run.suggest_float('factor_c', 0.75, 1.0, step=0.25),
            factor_t=run.suggest_categorical('factor_t', [0.75, 1.0, 1.5]),
            pool=t.cast(t.Any, run.suggest_categorical('pool', ['max', 'avg'])),
            activ_fn=t.cast(t.Any, run.suggest_categorical('activ_fn', ['ReLU', 'SiLU', 'GELU', 'LeakyReLU'])),
        )
        return model, model._metric_valid_f1_score

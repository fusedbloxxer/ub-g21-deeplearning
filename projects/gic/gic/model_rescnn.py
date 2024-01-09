import typing as t
from typing import cast, Tuple
import torch.nn as nn
from torch import Tensor
from functools import partial
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger as Logger
from optuna.trial import Trial as Run
from torcheval.metrics import Metric

from .data_dataset import GICDataset
from .model_base import F1ScoreObjective
from .model_base import ClassifierArgs, ClassifierModule


############ Modules ############
def AutoActiv(activ_fn: t.Literal['SiLU', 'GELU', 'ReLU', 'LeakyReLU']):
    """Creates an activation function."""
    match activ_fn:
        case      'SiLU': return nn.SiLU()
        case      'GELU': return nn.GELU()
        case      'ReLU': return nn.ReLU()
        case 'LeakyReLU': return nn.LeakyReLU()
        case _: raise ValueError('activation function {} is not supported'.format(activ_fn))


def AutoNorm(C: int,
             H: int,
             W: int,
             groups: int,
             norm: t.Literal['batch', 'layer', 'group', 'instn', 'none']):
    """Creates a normalization module."""
    match norm:
        case  'none': return nn.Identity()
        case 'batch': return nn.BatchNorm2d(C)
        case 'layer': return nn.LayerNorm([C, H, W])
        case 'instn': return nn.InstanceNorm2d(C)
        case 'group': return nn.GroupNorm(groups, C)
        case _: raise ValueError('normalization operation {} is not supported'.format(norm))


def AutoDrop(dim: t.Literal['channel', 'spatial'], chance: float):
    """Creates a dropout layer."""
    match dim:
        case 'channel':
            return nn.Dropout(p=chance)
        case 'spatial':
            return nn.Dropout2d(p=chance)
        case _:
            raise ValueError('droput operation {} is not supported'.format(dim))


def AutoConv(order: str,
             C: int,
             norm_fn: t.Callable[[], nn.Module],
             activ_fn: t.Callable[[], nn.Module],
             drop_fn: t.Callable[[], nn.Module]):
    """Creates a layer composed of: conv, norm, activ and dropout in a certain order."""
    operations = nn.ModuleDict({
        '0': nn.Conv2d(C, C, 3, 1, 1),
        '1': norm_fn(),
        '2': activ_fn(),
    })
    return nn.Sequential(
        *[operations[i] for i in order.split(' ')],
        drop_fn(),
    )


def AutoPool(reduce: t.Literal['max', 'avg', 'conv'],
             C: int, CO: int):
    """Creates a pooling layer followed by a convolution."""
    match reduce:
        case  'max': reduce_op = nn.MaxPool2d(2, 2, 0)
        case  'avg': reduce_op = nn.AvgPool2d(2, 2, 0)
        case 'conv': reduce_op = nn.Conv2d(C, C, 2, 2)
        case _: raise ValueError('reduce operation {} is not supported'.format(reduce))
    return nn.Sequential(reduce_op, nn.Conv2d(C, CO, 3, 1, 1))


def AutoRepeat(factory: t.Callable[[], nn.Module], count: int):
    """Creates a repeated forward layer."""
    return nn.Sequential(*[factory() for _ in range(count)])


def AutoSkip(layer_fn: t.Callable[[], nn.Module],
             repeat_fn: t.Callable[[t.Callable[[], nn.Module]], nn.Module]):
    """Creates a repeated forward layer with a skip connection from the input to the output."""
    return ResidualModule(partial(repeat_fn, layer_fn))


class ResidualModule(nn.Module):
    def __init__(self, factory: t.Callable[[], nn.Module]) -> None:
        super(ResidualModule, self).__init__()
        self.layer: nn.Module = factory()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


############ Neural Network ############
class ResCNNArgs(t.TypedDict):
    chan: int
    activ_fn: t.Literal['SiLU', 'GELU', 'LeakyReLU', 'ReLU']
    norm_layer: t.Literal['group', 'batch', 'layer']
    repeat: int
    groups: int
    dropout: float
    drop_type: t.Literal['channel', 'spatial']
    reduce: t.Literal['max', 'avg', 'conv']
    dense: int
    dropout_dense: float
    conv_order: str
    layers: int


class ResCNN(nn.Module):
    def __init__(self,
                 chan: int,
                 activ_fn: t.Literal['SiLU', 'GELU', 'LeakyReLU', 'ReLU'],
                 norm_layer: t.Literal['group', 'batch', 'layer'],
                 repeat: int,
                 groups: int,
                 dropout: float,
                 drop_type: t.Literal['channel', 'spatial'],
                 reduce: t.Literal['max', 'avg', 'conv'],
                 dense: int,
                 layers: int,
                 dropout_dense: float,
                 conv_order: str,
                 **kwargs) -> None:
        super(ResCNN, self).__init__()

        # The network handles 64x64 images
        H, W = 64, 64

        # Create layer factories
        auto_pool   = partial(AutoPool, reduce)
        auto_repeat = partial(AutoRepeat, count=repeat)
        auto_activ  = partial(AutoActiv, activ_fn=activ_fn)
        auto_skip   = partial(AutoSkip, repeat_fn=auto_repeat)
        auto_drop   = partial(AutoDrop, dim=drop_type, chance=dropout)
        auto_norm   = partial(AutoNorm, groups=groups, norm=norm_layer)
        auto_conv   = partial(AutoConv, order=conv_order, activ_fn=auto_activ, drop_fn=auto_drop)

        # Create the initial layer of the network
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, chan, 7, 1, 3),
            auto_norm(chan, H, W),
            auto_activ(),
        )

        # Create the convolutional backbone used to extract visual features
        self.cnn = nn.Sequential()
        for _ in range(layers):
            # Customize the factories for the current iteration
            norm = partial(auto_norm, C=chan, H=H, W=W)
            conv = partial(auto_conv, C=chan, norm_fn=norm)
            skip = partial(auto_skip, layer_fn=conv)

            # Construct residual block
            residual_module = nn.Sequential(
                skip(),
                auto_pool(chan, chan * 2),
                auto_norm(chan * 2, H // 2, W // 2),
                auto_activ(),
            )

            # Add module and prepare for next layer
            self.cnn.append(residual_module)

            # Spatial dimension is reduced while the depth is increased
            H, W, chan = H // 2, W // 2, chan * 2

        # Create the MLP final layer to classify images
        self.linear = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),

            nn.Linear(chan, dense),
            nn.BatchNorm1d(dense),
            nn.SiLU(),
            nn.Dropout(dropout_dense),

            nn.Linear(dense, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.cnn(x)
        x = self.linear(x)
        return x


############ Classification Module ############
class ResCNNClassifierArgs(ClassifierArgs, ResCNNArgs):
    pass


class ResCNNClassifier(ClassifierModule):
    def __init__(self, **kwargs: t.Unpack[ResCNNClassifierArgs]) -> None:
        super(ResCNNClassifier, self).__init__(name=cast(t.Any, kwargs.pop(cast(t.Any, 'name'), 'ResCNN')), **kwargs)
        self.rescnn = ResCNN(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.rescnn(x)


############ Objective ############
class ResCNNObjective(F1ScoreObjective):
    def __init__(self,
                 batch_size: int,
                 epochs: int,
                 data_module: t.Callable[[], LightningDataModule],
                 logger_fn: partial[Logger]):
        super(ResCNNObjective, self).__init__('ResCNN', batch_size, epochs, data_module, logger_fn)

    def model(self, run: Run) -> Tuple[LightningModule, Metric[Tensor]]:
        model = ResCNNClassifier(
            augment=run.suggest_categorical('augment', [True]),
            augment_n=run.suggest_categorical('augment_n', [1, 2, 3]),
            augment_m=run.suggest_categorical('augment_m', [11, 5, 4, 9, 7]),
            lr=run.suggest_float('lr', low=8e-5, high=6e-4),
            weight_decay=run.suggest_float('weight_decay', 1e-6, 6e-3),
            num_classes=GICDataset.num_classes,
            groups=run.suggest_categorical('groups', [2, 4, 8]),
            norm_layer=cast(t.Any, run.suggest_categorical('norm_layer', ['batch', 'group', 'layer'])),
            drop_type=cast(t.Any, run.suggest_categorical('drop_type', ['spatial', 'channel'])),
            repeat=run.suggest_int('repeat', 1, 4, step=1),
            dropout=run.suggest_float('dropout', 0.1, 0.3),
            layers=run.suggest_int('layers', 1, 3, step=1),
            chan=run.suggest_categorical('chan', [32, 64, 96, 128]),
            dense=run.suggest_categorical('dense', [128, 256, 384]),
            reduce=cast(t.Any, run.suggest_categorical('reduce', ['max', 'avg'])),
            dropout_dense=run.suggest_float('dropout_dense', 0.2, 0.5),
            activ_fn=cast(t.Any, run.suggest_categorical('activ_fn', ['SiLU', 'LeakyReLU', 'GELU', 'ReLU'])),
            conv_order=run.suggest_categorical('conv_order', ["0 1 2", "2 1 0", "0 2 1", "1 0 2", "2 0 1", "1 2 0"])
        )
        return model, model._metric_valid_f1_score

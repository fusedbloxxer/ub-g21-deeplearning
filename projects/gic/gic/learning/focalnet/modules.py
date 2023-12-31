import typing as t
import torch
import torch.nn as nn
from torch import Tensor


class AutoActivFnChoices(t.TypedDict):
     activ_fn: t.Literal['SiLU', 'GELU', 'ReLU', 'LeakyReLU']


def AutoActivFn(activ_fn: t.Literal['SiLU', 'GELU', 'ReLU', 'LeakyReLU'],
                **kwargs):
    return nn.ModuleDict({
        'SiLU': nn.SiLU(),
        'GELU': nn.GELU(),
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
    })[activ_fn]


class AutoNormChoices(t.TypedDict):
    C: int
    H: int
    W: int
    groups: int
    norm: t.Literal['batch', 'layer', 'group', 'instn', 'none']


def AutoNorm(C: int,
             H: int,
             W: int,
             groups: int,
             norm: t.Literal['batch', 'layer', 'group', 'instn', 'none'],
             **kwargs):
        return nn.ModuleDict({
            'none': nn.Identity(),
            'batch': nn.BatchNorm2d(C),
            'layer': nn.LayerNorm([C, H, W]),
            'instn': nn.InstanceNorm2d(C),
            'group': nn.GroupNorm(groups, C),
        })[norm]


class AutoDropoutChoices(t.TypedDict):
    dim: t.Literal['channel', 'spatial']
    probability: float


def AutoDropout(dim: t.Literal['channel', 'spatial'],
                probability: float,
                **kwargs):
    return nn.ModuleDict({
        'channel': nn.Dropout2d(p=probability),
        'spatial': nn.Dropout2d(p=probability),
    })[dim]


class ActivNormDropChoices(AutoNormChoices, AutoActivFnChoices, AutoDropoutChoices):
    pass


class AutoConvChoices(t.TypedDict):
    order: str


def AutoConv(order: str,
             **kwargs: t.Unpack[ActivNormDropChoices]):
    options = nn.ModuleDict({
        '0': nn.Conv2d(kwargs['C'], kwargs['C'], 3, 1, 1),
        '1': AutoNorm(**kwargs),
        '2': AutoActivFn(**kwargs),
    })
    modules = nn.Sequential(
        *[options[i] for i in order.split(' ')],
        AutoDropout(**kwargs),
    )
    return modules


class AutoReduceChoices(t.TypedDict):
    reduce: str


def AutoReduce(reduce: str,
               C: int,
               CO: int,
               **kwargs):
    reduce_op = nn.ModuleDict({
        'max': nn.MaxPool2d(2, 2, 0),
        'avg': nn.AvgPool2d(2, 2, 0),
        'conv': nn.Conv2d(C, C, 2, 2),
    })[reduce]
    channel_op = nn.Conv2d(C, CO, 3, 1, 1)
    return nn.Sequential(reduce_op, channel_op)


class RepeatModuleChoices(t.TypedDict):
    count: int


class RepeatModule(nn.Module):
    def __init__(self,
                 factory: t.Callable[[], nn.Module],
                 count: int = 1,
                 **kwargs) -> None:
        super(RepeatModule, self).__init__()
        self.layers = nn.Sequential(*[factory() for i in range(count)])

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResidualModuleChoices(t.TypedDict):
    pass


class ResidualModule(nn.Module):
    def __init__(self,
                 factory: t.Callable[[], nn.Module],
                 **kwargs) -> None:
        super(ResidualModule, self).__init__()
        self.layer = factory()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class AutoResidualChoices(ResidualModuleChoices, RepeatModuleChoices, ActivNormDropChoices, AutoConvChoices):
    pass


def AutoResidual(**kwargs: t.Unpack[AutoResidualChoices]):
    return ResidualModule(lambda: RepeatModule(lambda: AutoConv(**kwargs), **kwargs))


class FocalNetArgs(t.TypedDict):
    chan: int
    activ_fn: str
    norm_layer: str
    repeat: int
    groups: int
    dropout: float
    drop_type: str
    reduce: str
    dense: int
    dropout_dense: float
    conv_order: str
    layers: int


class FocalNetModule(nn.Module):
    def __init__(self,
                 chan: int,
                 activ_fn: str,
                 norm_layer: str,
                 repeat: int,
                 groups: int,
                 dropout: float,
                 drop_type: str,
                 reduce: str,
                 dense: int,
                 layers: int,
                 dropout_dense: float,
                 conv_order: str,
                 **kwargs) -> None:
        super(FocalNetModule, self).__init__()

        # Typehint casts
        activ_fn = t.cast(t.Literal['SiLU', 'GELU', 'LeakyReLU', 'ReLU'], activ_fn)
        norm_layer = t.cast(t.Literal['group', 'batch', 'layer'], norm_layer)
        drop_type = t.cast(t.Literal['channel', 'spatial'], drop_type)

        # Input Size
        H, W = 64, 64

        # RootLayer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, chan, 7, 1, 3),
            AutoNorm(chan, H, W, groups, norm_layer),
            AutoActivFn(activ_fn),
        )

        # ConvLayers
        self.cnn = nn.Sequential()
        for _ in range(layers):
            # Construct residual block
            residual_module = nn.Sequential(
                AutoResidual(
                    C=chan,
                    H=H,
                    W=W,
                    groups=groups,
                    norm=norm_layer,
                    activ_fn=activ_fn,
                    dim=drop_type,
                    probability=dropout,
                    order=conv_order,
                    count=repeat),
                AutoReduce(reduce, chan, chan * 2),
                AutoNorm(chan * 2, H // 2, W // 2, groups, norm_layer),
                AutoActivFn(activ_fn),
            )

            # Add module and prepare for next layer
            self.cnn.append(residual_module)
            H, W, chan = H // 2, W // 2, chan * 2

        # MLPLayer
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

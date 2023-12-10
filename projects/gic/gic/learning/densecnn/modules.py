import torch.nn as nn
import torch
from torch import Tensor
import typing as t


def create_activ_fn(variant: str):
    return nn.ModuleDict({
        'ReLU': nn.ReLU(),
        'SiLU': nn.SiLU(),
        'GELU': nn.GELU(),
        'LeakyReLU': nn.LeakyReLU(),
    })[variant]


def create_pool(variant: str):
    return nn.ModuleDict({
        'avg': nn.AvgPool2d(3, 2, 1),
        'max': nn.MaxPool2d(3, 2, 1),
    })[variant]


class DenseConvBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 features: int,
                 kernel: int,
                 pad: int,
                 activ_fn: str,
                 drop=0.3,
                 ) -> None:
        super(DenseConvBlock, self).__init__()
        self.bn_layer = nn.BatchNorm2d(ichan)
        self.activ_fn = create_activ_fn(activ_fn)
        self.conv_layer = nn.Conv2d(ichan, features, kernel, 1, pad, bias=True)
        self.drop = nn.Dropout2d(drop)

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
                 activ_fn: str,
                 drop=0.3
                 ) -> None:
        super(DenseCompConvBlock, self).__init__()
        assert 0 <= factor_c <= 1, 'invalid factor_c, must be in [0, 1]'

        # Dynamically sized block
        self.layers = nn.Sequential()

        # Add bottleneck
        if factor_c < 1:
            self.layers.add_module('comp_block', DenseConvBlock(ichan, int(ichan * factor_c), 1, 0, activ_fn, drop))

        # Add convolution
        self.layers.add_module('conv_block', DenseConvBlock(int(ichan * factor_c), features, 3, 1, activ_fn, drop))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 features: int,
                 factor_c: float,
                 activ_fn: str,
                 repeat: int,
                 drop=0.3,
                 ) -> None:
        super(DenseBlock, self).__init__()

        # Dynamically sized block
        self.layers = nn.ModuleList()

        # Add multiple inner blocks
        for l in range(repeat):
            self.layers.append(DenseCompConvBlock(ichan + l * features, features, factor_c, activ_fn, drop))

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
                x = torch.cat([x, l(x)], dim=1)
        return x


class DenseDownBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 factor_t: float,
                 activ_fn: str,
                 pool: str,
                 drop=0.3,
                 ) -> None:
        super(DenseDownBlock, self).__init__()

        # Reduce the number of input channels
        self.down_conv = DenseConvBlock(ichan, int(ichan * factor_t), 1, 0, activ_fn, drop)

        # Reduce the spatial dimension
        self.down_size = create_pool(pool)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_size(self.down_conv(x))


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
                 dense: int=128,
                 features: int=24,
                 factor_c: float=1.0,
                 factor_t: float=1.0,
                 activ_fn: t.Literal['ReLU', 'SiLU', 'GELU', 'LeakyReLU']='SiLU',
                 pool: t.Literal['max', 'avg']='max',
                 repeat: int=3,
                 inner: int=3,
                 f_drop=0.2,
                 c_drop=0.1,
                 **kwargs,
                 ) -> None:
        super().__init__()

        # Allow initial layer to extract multiple features
        chan = int(3 * features * 2)
        self.input_conv = nn.Conv2d(3, chan, 1, 1, 0)

        # Chain multiple blocks
        self.dense_layers = nn.Sequential()
        for _ in range(repeat):
            self.dense_layers.append(DenseBlock(chan, features, factor_c, activ_fn, inner, c_drop))
            chan += features * inner
            self.dense_layers.append(DenseDownBlock(chan, factor_t, activ_fn, pool, c_drop))
            chan = int(chan * factor_t)
        self.dense_layers.append(nn.AdaptiveMaxPool2d(1))
        self.dense_layers.append(nn.Flatten())

        # Apply classification
        self.classifier = nn.Sequential(
            nn.Dropout1d(f_drop),
            nn.Linear(in_features=chan, out_features=dense),
            nn.BatchNorm1d(dense),
            create_activ_fn(activ_fn),

            nn.Dropout1d(f_drop),
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


import torch.nn as nn
import torch
from torch import Tensor
import typing as t
from torch.profiler import record_function

from ..tune import HyperParameterSampler
from .autoencoder import AutoEncoder
from .modules import ConvBlock, ActivFn, create_activ_fn


class SpatialBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 ochan: int,
                 **kwargs) -> None:
        super(SpatialBlock, self).__init__()

        # Use multiple reduction paths
        self.layers = nn.ModuleList()
        self.layers.append(nn.AvgPool2d(3, 2, 1))
        self.layers.append(nn.MaxPool2d(3, 2, 1))

        # Aggregate back all paths
        self.aggregate = nn.Conv2d(ichan * 2, ochan, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Apply paths independently
        x = torch.cat([l(x) for l in self.layers], dim=1)

        # Combine all
        x = self.aggregate(x)
        return x


class PatchBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 ochan: int,
                 k: int,
                 p: int = 0,
                 reduce: bool = False,
                 **kwargs) -> None:
        super(PatchBlock, self).__init__()
        self.reduce = reduce

        # Extract patches by having kernel equal to stride
        self.patch_layer = nn.Conv2d(ichan, ochan, k, k, p, bias=True)

        # Optionally further downsample the image
        if reduce:
            self.reduce_layer = SpatialBlock(ochan, ochan)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_layer(x)

        if self.reduce:
            x = self.reduce_layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(self,
                 module_factory: t.Callable[[], nn.Module],
                 **kwargs) -> None:
        super(ResBlock, self).__init__()

        # Create Inner Layer
        self.layer = module_factory()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


class DenseBlock(nn.Module):
    def __init__(self,
                 repeat: int,
                 module_factory: t.Callable[[], nn.Module],
                 **kwargs) -> None:
        super(DenseBlock, self).__init__() # TODO

        # Create Inner Layers
        self.layers = nn.ModuleList([module_factory() for _ in range(repeat)])

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = torch.cat([x, l(x)], dim=1)
        return x


class ConvNextNet(nn.Module):
    def __init__(self,
                 chan: int,
                 patch: int,
                 h: int = 64,
                 w: int = 64,
                 activ_fn: ActivFn = 'GELU',
                 patch_reduce: bool = False,
                 conv_dropout: float = 0.0,
                 conv_layers: int = 1,
                 dense_dropout: float = 0.0,
                 dense_features: int = 128,
                 dense_layers: int = 1,
                 **kwargs) -> None:
        super(ConvNextNet, self).__init__()
        assert 0 <= dense_layers <= 2, 'dense_layers must be in [0, 2]'
        assert 1 <= conv_layers <= 3, 'conv-layers must be in [1, 3]'

        # Internals
        self.__dense_dropout = dense_dropout
        self.__dense_features = dense_features
        self.__conv_dropout = conv_dropout
        self.__dense_layers = dense_layers
        self.__conv_layers = conv_layers
        self.__activ_fn: ActivFn = activ_fn

        # Extract Patches and Downsample
        self.input_layer = PatchBlock(3, chan, patch, reduce=patch_reduce)
        h = h // patch // (2 if patch_reduce else 1)
        w = w // patch // (2 if patch_reduce else 1)

        # Convolutional Layers
        self.conv_layers, chan = self.__create_con_layers(chan, h, w)

        # Classification Layers
        self.classifier = nn.Sequential(
            self.__create_fcn_layers(chan),
            nn.Linear(in_features=dense_features, out_features=100)
        )

    def __create_con_layers(self, chan: int, h: int, w: int):
        # Transform using Convolutional Layers
        conv_layers = nn.Sequential()

        for _ in range(self.__conv_layers):
            conv_layers.append(nn.Sequential(
                nn.Dropout2d(self.__conv_dropout),
                ResBlock(lambda: ConvBlock(chan, h, w, chan * 4, self.__activ_fn)),
                SpatialBlock(chan, chan * 2),
            ))
            chan *= 2
            w //= 2
            h //= 2

        conv_layers.append(nn.AdaptiveAvgPool2d(1))
        conv_layers.append(nn.Flatten())
        return conv_layers, chan

    def __create_fcn_layers(self, chan: int):
        # Transform using Fully-Connect Layers
        fcn_layers = nn.Sequential()

        # Project from conv_layers to linear_layers
        fcn_layers.append(nn.Sequential(
            nn.Dropout(self.__dense_dropout),
            nn.Linear(in_features=chan, out_features=self.__dense_features),
            nn.BatchNorm1d(self.__dense_features),
            create_activ_fn(self.__activ_fn)
        ))

        # Additional layers
        for _ in range(self.__dense_layers):
            fcn_layers.append(nn.Sequential(
                nn.Dropout(self.__dense_dropout),
                ResBlock(lambda: nn.Sequential(
                    nn.Linear(in_features=self.__dense_features, out_features=self.__dense_features),
                    nn.BatchNorm1d(self.__dense_features),
                    create_activ_fn(self.__activ_fn),
                )),
            ))
        return fcn_layers

    def forward(self, x: Tensor) -> Tensor:
        # Extract local information from the image
        x = self.input_layer(x)
        x = self.conv_layers(x)

        # Concatenate and predict
        x = self.classifier(x)
        return x


convnext_sampler = HyperParameterSampler(lambda trial: {
    'epochs': 150,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'patch': 4,
    'lr': trial.suggest_float('lr', 9e-5, 9e-4),
    'weight_decay': trial.suggest_float('weight_decay', 4e-6, 1e-4),
    'chan': trial.suggest_int('chan', 32, 96, step=32),
    'conv_dropout': trial.suggest_float('conv_dropout', 0.2, 0.2),
    'conv_layers': trial.suggest_int('conv_layers', 1, 3, step=1),
    'dense_dropout': trial.suggest_float('dense_dropout', 0.1, 0.3),
    'dense_features': trial.suggest_int('dense_features', 128, 512, step=128),
    'dense_layers': trial.suggest_int('dense_layers', 0, 2, step=1),
    'activ_fn': trial.suggest_categorical('activ_fn', ['SiLU', 'GELU']),
    'patch_reduce': trial.suggest_categorical('patch_reduce', [True, False]),
})

import torch.nn as nn
from torch import Tensor
import typing as t

from ..tune import HyperParameterSampler


class ResConvBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, dropout: float, activ_fn: nn.Module) -> None:
        super(ResConvBlock, self).__init__()

        self.drop = nn.Dropout2d(p=dropout)
        self.aux_layer = nn.Conv2d(
            in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv_layer1 = nn.Conv2d(
            in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(
            out_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.activ_fn = activ_fn
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aux_layer(x)
        x2 = self.activ_fn(self.conv_layer1(x))
        x2 = self.activ_fn(self.conv_layer2(x2))
        x2 = self.bn(x2)
        x2 = self.drop(x2)
        return x1 + x2


class ResCNN(nn.Module):
    def __init__(self,
                 pool: t.Literal['max', 'avg'],
                 dropout1d: float = 0.4,
                 dropout2d: float = 0.2,
                 conv_chan: int = 32,
                 dens_chan: int = 512,
                 activ_fn: str = 'ReLU',
                 **kwargs) -> None:
        super(ResCNN, self).__init__()

        self.activ_fn = getattr(nn, activ_fn)()
        self.pool = nn.MaxPool2d(2, 2) if pool == 'max' else nn.AvgPool2d(2, 2)
        self.adapative_pool = nn.AdaptiveMaxPool2d(
            1) if pool == 'max' else nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Conv2d(3, conv_chan, 3, 1, 1),
            self.activ_fn,
            nn.BatchNorm2d(conv_chan),
            nn.Dropout2d(dropout2d),
            self.pool,
            ResConvBlock(conv_chan, conv_chan * 2, dropout2d, self.activ_fn),
            self.pool,
            ResConvBlock(conv_chan * 2, conv_chan *
                         4, dropout2d, self.activ_fn),
            self.pool,
            ResConvBlock(conv_chan * 4, conv_chan *
                         8, dropout2d, self.activ_fn),
            self.adapative_pool,
            nn.Flatten(),
            nn.Linear(in_features=conv_chan * 8, out_features=dens_chan),
            self.activ_fn,
            nn.BatchNorm1d(dens_chan),
            nn.Dropout(dropout1d),
            nn.Linear(in_features=dens_chan, out_features=dens_chan),
            self.activ_fn,
            nn.BatchNorm1d(dens_chan),
            nn.Dropout(dropout1d),
            nn.Linear(in_features=dens_chan, out_features=100),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

resnet_sampler = HyperParameterSampler(lambda trial: {
    'batch_size': trial.suggest_int('batch_size', 16, 32, step=16),
    'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
    'lr': trial.suggest_float('lr', 1e-4, 4e-3),
    'epochs': trial.suggest_int('epochs', 60, 100),
    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
    'pool': trial.suggest_categorical('pool', ['max', 'avg']),
    'dropout1d': trial.suggest_float('dense_dropout', 0.2, 0.6),
    'dropout2d': trial.suggest_float('conv_dropout', 0.3, 0.6),
    'conv_chan': trial.suggest_int('conv_chan', 16, 32, step=8),
    'dens_chan': trial.suggest_int('dens_chan', 128, 512, 128),
    'activ_fn': trial.suggest_categorical('activ', ['ReLU', 'SiLU', 'GELU'])
})

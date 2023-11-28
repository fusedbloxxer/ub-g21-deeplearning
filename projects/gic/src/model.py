import torch.nn as nn
import torch
from torch import Tensor
import typing as t


class ResConvBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super(ResConvBlock, self).__init__()

        self.drop = nn.Dropout2d(p=0.4)
        self.aux_layer = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv_layer1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.activ_fn = nn.SiLU()
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aux_layer(x)
        x2 = self.activ_fn(self.conv_layer1(x))
        x2 = self.activ_fn(self.conv_layer2(x2))
        x2 = self.bn(x2)
        x2 = self.drop(x2)
        return x1 + x2


class ResCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ResCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResConvBlock(in_chan=64, out_chan=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResConvBlock(in_chan=256, out_chan=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResConvBlock(in_chan=512, out_chan=1024),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout1d(p=0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=100),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

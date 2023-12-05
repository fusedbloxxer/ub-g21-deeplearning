import os as so
import sys as s
import pathlib as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader, ConcatDataset
import torcheval
from torcheval.metrics import MulticlassF1Score, Mean
import optuna as opt
import torchvision as tn
import sklearn as sn
from sklearn.metrics import f1_score
import pandas as ps
import numpy as ny
import typing as t
import pathlib as pl
import matplotlib.pyplot as pt
import random as rng
from tqdm import tqdm
import tqdm as tm
from pprint import pprint
from git import Repo
import lightning as tl
import kornia as K
from typing import Any
from torch.nn.functional import mse_loss
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from gic import *
from gic.models.modules import ConvBlock, ActivFn


class DownSampleBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs
                 ) -> None:
        super(DownSampleBlock, self).__init__()
        self.conv_layer = ConvBlock(ichan, H, W, hchan, activ_fn)
        self.down_layer = nn.Conv2d(ichan, ichan * 2, 2, 2, 0, bias=False)

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        x0: Tensor = self.conv_layer(x)
        x1: Tensor = self.down_layer(x0)
        return x1, x0


class UpSampleBlock(nn.Module):
    def __init__(self,
                 ichan: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 up: bool = True,
                 **kwargs
                 ) -> None:
        super(UpSampleBlock, self).__init__()
        self.up = up
        self.conv_layer = ConvBlock(ichan * 2, H, W, hchan, activ_fn)

        if self.up:
            self.up_layer = nn.ConvTranspose2d(ichan * 2, ichan // 2, 2, 2, 0, bias=False)
        else:
            self.chan_layer = nn.Conv2d(ichan * 2, ichan, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layer(x)
        x = self.up_layer(x) if self.up else self.chan_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 ichan: int,
                 chan: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs) -> None:
        super(Encoder, self).__init__()

        # Layers
        self.in_layer = nn.Conv2d(ichan, chan, 3, 1, 1)
        self.layers = nn.ModuleList([
            DownSampleBlock(chan * 1, H // 1, W // 1, hchan * 1, activ_fn),
            DownSampleBlock(chan * 2, H // 2, W // 2, hchan * 2, activ_fn),
            DownSampleBlock(chan * 4, H // 4, W // 4, hchan * 4, activ_fn),
            DownSampleBlock(chan * 8, H // 8, W // 8, hchan * 8, activ_fn),
        ])

    def forward(self, x: Tensor) -> t.Tuple[Tensor, t.List[Tensor]]:
        # Prepare input
        x = self.in_layer(x)

        # Downsampling
        activ_maps: t.List[Tensor] = []
        for layer in self.layers:
            output: t.Tuple[Tensor, Tensor] = layer(x)
            x, activ_map = output
            activ_maps.append(activ_map)
        return x, activ_maps


class Decoder(nn.Module):
    def __init__(self,
                 chan: int,
                 ochan: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs) -> None:
        super(Decoder, self).__init__()

        # Layers
        self.layers = nn.ModuleList([
            UpSampleBlock(chan * 8, H // 8, W // 8, hchan * 8, activ_fn),
            UpSampleBlock(chan * 4, H // 4, W // 4, hchan * 4, activ_fn),
            UpSampleBlock(chan * 2, H // 2, W // 2, hchan * 2, activ_fn),
            UpSampleBlock(chan * 1, H // 1, W // 1, hchan * 1, activ_fn, up=False),
        ])
        self.out_layer = nn.Conv2d(chan, ochan, 3, 1, 1)

    def forward(self, x: Tensor, activ_maps: t.List[Tensor]) -> Tensor:
        # Upsampling
        for i, layer in enumerate(self.layers):
            o = activ_maps[len(self.layers) - i - 1]
            x = torch.cat([o, x], dim=1)
            x = layer(x)

        # Prepare output
        x = self.out_layer(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,
                 ichan: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs
                 ) -> None:
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(ichan, H, W, hchan, activ_fn),
            nn.ConvTranspose2d(ichan, ichan // 2, 2, 2, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self,
                 chan: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs) -> None:
        super(UNet, self).__init__()

        self.encoder = Encoder(3, chan, 64, 64, hchan, activ_fn)
        self.bottleneck = Bottleneck(chan * 16, 64 // 16, 64 // 16, hchan * 16, activ_fn)
        self.decoder = Decoder(chan, 3, 64, 64, hchan, activ_fn)

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        x, activ_maps = self.encoder(x)
        b = x = self.bottleneck(x)
        x = self.decoder(x, activ_maps)
        return x, b


class AutoEncoder(tl.LightningModule):
    def __init__(self,
                 chan: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs) -> None:
        super(AutoEncoder, self).__init__()
        self.unet = UNet(chan, hchan, activ_fn)

    def forward(self, inputs):
        return self.unet(inputs)

    def on_train_start(self) -> None:
        # self.mask = MaskAugment()
        # self.norm = GenImageAugment(augment=False, normalize=True)
        pass

    def training_step(self, b: Tensor, idx: int) -> STEP_OUTPUT:
        X_true, _ = b
        X_mask = self.norm(self.mask(X_true))
        X_true = self.norm(X_true)
        X_fake, _ = self(X_mask)
        loss = mse_loss(X_fake, X_true)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=9e-4)

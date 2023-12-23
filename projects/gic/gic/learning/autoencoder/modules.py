import typing as t
from typing import Tuple
import torch
import torch.nn as nn
import pathlib as pl
from torch import Tensor
from collections import OrderedDict


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
                self.layers.append(nn.Conv2d(ichan, ichan, 3, 1, padding='same'))
            case _:
                raise ValueError('input dimension {} is not supported'.format(idim))

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
                self.layers.append(nn.Conv2d(ochan, ochan, 3, 1, padding='same'))
            case _:
                raise ValueError('output dimension {} is not supported'.format(idim))

        # Skip Layer
        dim  = 0
        dim += 1 if idim == 'expand' else -1 if idim == 'reduce' else 0
        dim += 1 if odim == 'expand' else -1 if odim == 'reduce' else 0
        self.skip = nn.Sequential()

        # Adjust spatial dimension
        if   dim > 0:
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
            ConvBlock(       3, chan * 1, 64, 64, odim='reduce'),
            ConvBlock(chan * 1, chan * 2, 32, 32, odim='reduce'),
            ConvBlock(chan * 2, chan * 4, 16, 16, odim='reduce'),
            ConvBlock(chan * 4,   latent,  8,  8, odim='reduce'),
        )

        self.decoder = nn.Sequential(
            ConvBlock(  latent, chan * 4,  8,  8, idim='expand'),
            ConvBlock(chan * 4, chan * 2, 16, 16, idim='expand'),
            ConvBlock(chan * 2, chan * 1, 32, 32, idim='expand'),
            ConvBlock(chan * 1,        3, 64, 64, idim='expand', out_activ_fn=False),
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
                 dropout: float | None=None,
                 batch: str | None=None,
                 **kwargs) -> None:
        super(DAEClassifier, self).__init__()

        # Use pretrained denoising autoencoder
        self.autoencoder = ConvAutoEncoder(64, 64)
        self.autoencoder.load_state_dict(OrderedDict(list(map(lambda x: (x[0].replace('autoencoder.', ''), x[1]), torch.load(str(ckpt_path / 'dae.pt')).items()))))

        # Choose an activation
        activ = nn.ModuleDict({ 'silu': nn.SiLU(), 'leak': nn.LeakyReLU() })[activ_fn]

        # Train a classifier on top
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(64, dense),
            nn.BatchNorm1d(dense) if batch else nn.Identity(),
            activ,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(dense, dense),
            nn.BatchNorm1d(dense) if batch else nn.Identity(),
            activ,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(dense, 100),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.autoencoder.encode(x)
        x = self.classifier.forward(x)
        return x

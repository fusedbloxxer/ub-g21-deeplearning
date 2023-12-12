import typing as t
import torch
import torch.nn as nn
from torch import Tensor


class ConvModule(nn.Module):
    def __init__(self, hchan: int, h: int, w: int, p: float, **kwargs) -> None:
        super(ConvModule, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.conv_layer = nn.Conv2d(hchan, hchan, 3, 1, 1)
        self.norm_layer = nn.LayerNorm([hchan, h, w])
        self.actv_layer = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        q: Tensor = self.dropout(x)
        q = self.conv_layer(q)
        q = self.norm_layer(q)
        q = self.actv_layer(q)
        return x + q


class InputModule(nn.Module):
    def __init__(self, ichan: int, hchan: int, h: int, w: int, p: float, **kwargs) -> None:
        super(InputModule, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.conv_layer = nn.Conv2d(ichan, hchan, 3, 1, 1)
        self.norm_layer = nn.LayerNorm([hchan, h, w])
        self.actv_layer = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.actv_layer(x)
        return x


class EncodeModule(nn.Module):
    def __init__(self, hchan: int, ochan: int, h: int, w: int, p: float, **kwargs) -> None:
        super(EncodeModule, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.pool_layer = nn.MaxPool2d((2, 2), 2)
        self.conv_layer = nn.Conv2d(hchan, ochan, 3, 1, 1)
        self.norm_layer = nn.LayerNorm([ochan, h // 2, w // 2])
        self.actv_layer = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.pool_layer(x)
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.actv_layer(x)
        return x


class DenoisingEncoder(nn.Module):
    def __init__(self, ichan: int, hchan: int, h: int, w: int, layers: int, drop_input: float, drop_inside: float, **kwargs) -> None:
        super(DenoisingEncoder, self).__init__()
        self.in_layer = InputModule(ichan, hchan, h, w, drop_input)
        self.layers = nn.Sequential()
        for l in range(layers):
            self.layers.append(ConvModule(hchan * 2**l, h // 2**l, w // 2**l, drop_inside))
            self.layers.append(EncodeModule(hchan * 2**l, hchan * 2**(l+1), h // 2**l, w // 2**l, drop_inside))

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        x = self.layers(x)
        return x


class DecodeModule(nn.Module):
    def __init__(self, hchan: int, ochan: int, h: int, w: int, p: float, **kwargs) -> None:
        super(DecodeModule, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.conv_layer = nn.ConvTranspose2d(hchan, ochan, 3, 1, 1)
        self.pool_layer = nn.Upsample(scale_factor=2)
        self.norm_layer = nn.LayerNorm([ochan, h * 2, w * 2])
        self.actv_layer = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.conv_layer(x)
        x = self.pool_layer(x)
        x = self.norm_layer(x)
        x = self.actv_layer(x)
        return x


class OutputModule(nn.Module):
    def __init__(self, hchan: int, ochan: int, h: int, w: int, **kwargs) -> None:
        super(OutputModule, self).__init__()
        self.conv_layer = nn.Conv2d(hchan, ochan, 3, 1, 1)
        self.actv_layer = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layer(x)
        x = self.actv_layer(x)
        return x


class DenoisingDecoder(nn.Module):
    def __init__(self, hchan: int, ochan: int, h: int, w: int, layers: int, drop_inside: float, **kwargs) -> None:
        super(DenoisingDecoder, self).__init__()
        self.layers = nn.Sequential()
        for l in range(layers - 1, -1, -1):
            self.layers.append(DecodeModule(hchan * 2**(l+1), hchan * 2**l, h // 2**(l+1), w // 2**(l+1), drop_inside))
            self.layers.append(ConvModule(hchan * 2**l, h // 2**l, w // 2**l, drop_inside))
        self.out_layer = OutputModule(hchan, ochan, h, w)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.out_layer(x)
        return x


class DenoisingBottleneck(nn.Module):
    def __init__(self, ichan: int, hchan: int, ochan: int, h: int, w: int) -> None:
        super(DenoisingBottleneck, self).__init__()
        self.in_layer = nn.Conv2d(ichan, hchan, (h, w), (h, w), 0)
        self.norm_layer_1 = nn.LayerNorm([hchan, 1, 1])
        self.out_layer = nn.ConvTranspose2d(hchan, ochan, (h, w), (h, w), 0)
        self.norm_layer_2 = nn.LayerNorm([ochan, h, w])
        self.activ_fn = nn.SiLU()

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        x = self.in_layer(x)
        x = self.norm_layer_1(x)
        l = x = self.activ_fn(x)
        x = self.out_layer(x)
        x = self.norm_layer_2(x)
        x = self.activ_fn(x)
        return x, l


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, ichan: int, hchan: int, ochan: int, lowdim: int, h: int, w: int, layers: int, drop_input: float, drop_inside: float, **kwargs) -> None:
        super(DenoisingAutoEncoder, self).__init__()
        assert ichan == ochan, 'the input and output number of channels must match'
        assert layers % 2 == 0 and layers >= 0, 'the number of layers must be evenly distributed'
        layers = layers // 2
        self.encoder = DenoisingEncoder(ichan, hchan, h, w, layers, drop_input, drop_inside)
        self.bottleneck = DenoisingBottleneck(hchan * 2**layers, lowdim, hchan * 2**layers, h // 2**layers, w // 2**layers)
        self.decoder = DenoisingDecoder(hchan, ochan, h, w, layers, drop_inside)

    def encode(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)[1]
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)[0]
        x = self.decoder(x)
        return x

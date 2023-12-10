import torch.nn as nn
import torch
from torch import Tensor
import typing as t

ActivFn = t.Literal['SiLU', 'GELU']


def create_activ_fn(variant: ActivFn):
    return nn.ModuleDict({
        'SiLU': nn.SiLU(),
        'GELU': nn.GELU(),
    })[variant]


class ConvBlock(nn.Module):
    def __init__(self,
                 C: int,
                 H: int,
                 W: int,
                 hchan: int,
                 activ_fn: ActivFn,
                 **kwargs) -> None:
        super(ConvBlock, self).__init__()
        assert hchan > C, f'pointwise layer should have more channels than the input {hchan} > {C}'

        # Apply multiple depthwise-convolution paths
        self.depthwise_layer = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C),
            nn.LayerNorm([C, H, W]),
        )

        # Aggregate feature maps from all inputs
        self.pointwise_layer = nn.Conv2d(C, hchan, 1, 1, 0)

        # Leverage non-linearities to understand complex functions
        self.activ_fn = create_activ_fn(activ_fn)

        # Bottleneck the result for faster processing
        self.bottleneck_layer = nn.Conv2d(hchan, C, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        x = self.activ_fn(x)
        x = self.bottleneck_layer(x)
        return x

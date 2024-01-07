import typing as t
import torch
import kornia as K
import kornia.augmentation as KA
from torch import Tensor
from torch.nn.functional import one_hot

from .data_dataset import GICDataset


class PreprocessTransform(object):
    def __init__(self, mean: Tensor=GICDataset.mean, std: Tensor=GICDataset.std) -> None:
        super(PreprocessTransform, self).__init__()
        self.__std = std
        self.__mean = mean

    def norm(self, x: Tensor) -> Tensor:
        return K.enhance.normalize(x, self.__mean, self.__std)

    def denorm(self, x: Tensor) -> Tensor:
        return K.enhance.denormalize(x, self.__mean, self.__std)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AugmentTransform(object):
    def __init__(self) -> None:
        super(AugmentTransform, self).__init__()
        self.__ops = KA.ImageSequential(
            KA.RandomHorizontalFlip(p=0.5),
            KA.RandomVerticalFlip(p=0.5),
            KA.RandomAffine(45, None, (0.85, 1.25))
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        return self.__ops(x)


class RobustAugmentTransform(object):
    def __init__(self,
                 rand_n: int=2,
                 rand_m: int=4,
                 flip_p: float=0.5,
                 cut_mix_p: float=0.5) -> None:
        super(RobustAugmentTransform, self).__init__()
        assert 0 < rand_n
        assert 0 < rand_m
        assert 0 <= flip_p <= 1
        assert 0 <= cut_mix_p <= 1
        self.__fipping = KA.ImageSequential(KA.RandomHorizontalFlip(p=flip_p), KA.RandomVerticalFlip(p=flip_p))
        self.__cut_mix = KA.RandomCutMixV2(data_keys=['input', 'class'], p=cut_mix_p)
        self.__rand = KA.auto.RandAugment(n=rand_n, m=rand_m)

    @torch.no_grad()
    def __call__(self, x: Tensor, y: Tensor) -> t.List[Tensor]:
        # Flip -> RandomAugment -> CutMix
        x = self.__fipping(x)
        x = self.__rand(x)
        x, y = self.__cut_mix(x, y)

        # Blend the output label between the original and the inserted mixup
        y = y.squeeze(0)
        y_origin = one_hot(y[:, 0].type(dtype=torch.int64), num_classes=GICDataset.num_classes)
        y_cutmix = one_hot(y[:, 1].type(dtype=torch.int64), num_classes=GICDataset.num_classes)
        y_factor = y[:, 2].unsqueeze(-1)
        y_blend = (1. - y_factor) * y_origin + y_factor * y_cutmix

        # Augmented images x An array of probabilities for each class
        return [x, y_blend]


class MaskingNoiseTransform(object):
    def __init__(self, p=0.65) -> None:
        super(MaskingNoiseTransform, self).__init__()
        self.p: float = p

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        return x * torch.bernoulli(torch.full((B, 1, H, W), fill_value=self.p, device=x.device)).tile((1, 3, 1, 1))

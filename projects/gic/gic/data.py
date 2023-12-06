from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import kornia as K
import kornia.augmentation as KA
from torch import Tensor
import torchvision as tn
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import Compose, ToDtype, Resize, Normalize
import typing as t
from typing import Any, Tuple, TypeAlias, Union, Literal, overload
import pathlib as pl
import pandas as ps
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import lightning as tl


TrainSplit: TypeAlias = Tuple[float, float] | Literal['joint', 'disjoint']


class GICDataset(data.Dataset[t.Tuple[Tensor, Tensor] | Tensor]):
    def __init__(self,
                 path: pl.Path,
                 split: t.Literal['train', 'val', 'test']) -> None:
        super(GICDataset, self).__init__()
        self.__root = path
        self.__split = split
        self.__meta = self.__root / f'{self.__split}.csv'
        self.__imag = self.__root / f'{self.__split}_images'
        self.__data = ps.read_csv(self.__meta)

    def __getitem__(self, index: int):
        image_name: str = self.__data.iloc[index].loc['Image']
        image_path: str = str(self.__imag / image_name)
        image: Tensor = tn.io.read_image(image_path, tn.io.ImageReadMode.RGB)
        image = F.to_dtype(image, torch.float, scale=True)

        if self.__split == 'test':
            return image

        label: Tensor = torch.tensor(self.__data.iloc[index].loc['Class'])
        return (image, label)

    def __len__(self) -> int:
        return len(self.__data)


class GICPreprocess(nn.Module):
    def __init__(self, augment=True, normalize=True, **kwargs) -> None:
        super(GICPreprocess, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if not augment and not normalize:
            self.__ops = nn.Identity()
        else:
            self.__ops = KA.ImageSequential()

        if augment:
            self.__ops.append(KA.auto.AutoAugment('cifar10'))
        if normalize:
            self.__ops.append(KA.Normalize(self.mean, self.std))

    def denorm(self, x: Tensor) -> Tensor:
        return K.enhance.denormalize(x, self.mean, self.std)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.__ops(x)


class GICPerturb(nn.Module):
    def __init__(self, mask=True, normalize=False, noise=0.0075, p=0.75) -> None:
        super(GICPerturb, self).__init__()
        self.__ops = K.augmentation.ImageSequential()
        self.__v_noise = noise
        self.__p_noise = p

        if mask:
            self.__ops.append(K.augmentation.ImageSequential(
                K.augmentation.ImageSequential(
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    K.augmentation.RandomErasing(scale=(0.01, 0.05), p=0.35),
                    same_on_batch=False,
                ),
                nn.Identity(),
                random_apply=1,
                same_on_batch=False,
                random_apply_weights=[0.75, 0.25],
            ))

        if normalize:
            self.__ops.append(KA.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    @torch.no_grad()
    def noise(self, x: Tensor) -> Tensor:
        p = torch.tensor(self.__p_noise, device=x.device)
        v = torch.tensor(self.__v_noise, device=x.device)
        s = torch.bernoulli(p)
        if s == 1:
            return torch.sqrt(v) * torch.randn_like(x, device=x.device)
        else:
            return torch.zeros_like(x, device=x.device)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.__ops(x + self.noise(x))


class GICDatasetModule(tl.LightningDataModule):
    def __init__(self,
                 path: pl.Path,
                 final: bool,
                 batch_sz: int,
                 workers: int,
                 prefetch: int | None,
                 pin: bool,
                 augment: bool,
                 gen: torch.Generator,
                 normalize: bool=True,
                 ) -> None:
        super().__init__()
        self.__pin = pin
        self.__path = path
        self.__final = final
        self.__gen = gen
        self.__workers = workers
        self.__batch_sz = batch_sz
        self.__prefetch = prefetch
        self.__tr_norm = GICPreprocess(augment=False, normalize=normalize)
        self.__tr_augm = GICPreprocess(augment=augment, normalize=normalize)

    def setup(self, stage: str):
        self.ds = load_data(self.__path, disjoint=not self.__final)
        self.train_ds, self.valid_ds, self.test_ds = self.ds

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.__batch_sz, num_workers=self.__workers, prefetch_factor=self.__prefetch, pin_memory=self.__pin, generator=self.__gen)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, shuffle=False, batch_size=self.__batch_sz, num_workers=self.__workers, prefetch_factor=self.__prefetch, pin_memory=self.__pin, generator=self.__gen)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.__batch_sz, num_workers=self.__workers, prefetch_factor=self.__prefetch, pin_memory=self.__pin, generator=self.__gen)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.__batch_sz, num_workers=self.__workers, prefetch_factor=self.__prefetch, pin_memory=self.__pin, generator=self.__gen)

    def on_after_batch_transfer(self, batch: Tensor, _: int) -> Any:
        if t.cast(tl.Trainer, self.trainer).training:
            batch[0] = self.__tr_augm(batch[0])
        elif t.cast(tl.Trainer, self.trainer).validating:
            batch[0] = self.__tr_norm(batch[0])
        elif isinstance(batch, (tuple, list)):
            batch[0] = self.__tr_norm(batch[0])
        else:
            batch = self.__tr_norm(batch)
        return batch


@overload
def load_data(path: pl.Path) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], None, data.Dataset[Tensor]]:
    ...
@overload
def load_data(path: pl.Path, disjoint: bool) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
    ...
def load_data(path: pl.Path, disjoint: bool = True) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], None, data.Dataset[Tensor]] | t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
    train_data = GICDataset(path, 'train')
    valid_data = GICDataset(path, 'val')
    test_data = GICDataset(path, 'test')

    test_data = t.cast(data.Dataset[Tensor], test_data)
    valid_data = t.cast(data.Dataset[t.Tuple[Tensor, Tensor]], valid_data)
    train_data = t.cast(data.Dataset[t.Tuple[Tensor, Tensor]], train_data)

    if not disjoint:
        train_data = data.ConcatDataset([train_data, valid_data])
        return train_data, None, test_data

    return train_data, valid_data, test_data


def load_batched_data(path: pl.Path, split: TrainSplit, gen: torch.Generator, **kwargs):
    # Load disjoint subsets
    train_ds, valid_ds, test_ds = load_data(path, True)

    # Decide how to split the training data
    if isinstance(split, tuple):
        train_ds = ConcatDataset([train_ds, valid_ds])
        train_ds, valid_ds = random_split(train_ds, split, gen)
    elif split == 'joint':
        train_ds = ConcatDataset([train_ds, valid_ds])
        valid_ds = None
    elif split == 'disjoint':
        pass

    # Create dataloaders for each subset
    train_dl: DataLoader[Tuple[Tensor, Tensor]] = DataLoader(train_ds, shuffle=True, generator=gen, **kwargs)
    valid_dl: DataLoader[Tuple[Tensor, Tensor]] | None = DataLoader(valid_ds, shuffle=False, generator=gen, **kwargs) if valid_ds else None
    test_dl: DataLoader[Tensor] = DataLoader(test_ds, shuffle=False, generator=gen, **kwargs)
    return train_dl, valid_dl, test_dl

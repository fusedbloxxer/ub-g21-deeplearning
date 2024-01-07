import torch
from torch import Tensor
import typing as t
from typing import Any, Literal, Tuple, TypeAlias, overload
import pathlib as pl
import torch.utils.data as data
from torch.utils.data import DataLoader, ConcatDataset, random_split

from .data_dataset import GICDataset


TrainSplit: TypeAlias = Tuple[float, float] | Literal['joint', 'disjoint']


@overload
def load_data(path: pl.Path) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], None, data.Dataset[Tensor]]:
    ...
@overload
def load_data(path: pl.Path, disjoint: bool) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
    ...
def load_data(path: pl.Path, disjoint: bool = True) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], None, data.Dataset[Tensor]] | t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
    train_data = GICDataset(path, 'train')
    valid_data = GICDataset(path, 'valid')
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

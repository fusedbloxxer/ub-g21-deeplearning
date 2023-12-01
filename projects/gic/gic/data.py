import torch
from torch import Tensor
import torchvision as tn
from torchvision.transforms.v2 import Compose, ToDtype, Resize, Normalize
import typing as t
from typing import Tuple, TypeAlias, Union, Literal
import pathlib as pl
import pandas as ps
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split

from . import pin_memory


TrainSplit: TypeAlias = Tuple[float, float] | Literal['disjoint', 'train']


class GenImageDataset(data.Dataset[t.Tuple[Tensor, Tensor] | Tensor]):
    def __init__(self,
                 path: pl.Path,
                 split: t.Literal['train', 'val', 'test'],
                 preprocess: bool = True) -> None:
        super(GenImageDataset, self).__init__()

        self.__split = split
        self.__root = path
        self.__meta = self.__root / f'{self.__split}.csv'
        self.__imag = self.__root / f'{self.__split}_images'

        self.__data = ps.read_csv(self.__meta)
        self.__preprocess = preprocess
        self.__transform = Compose([
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index: int):
        image_name: str = self.__data.iloc[index].loc['Image']
        image_path: str = str(self.__imag / image_name)
        image: Tensor = tn.io.read_image(image_path, tn.io.ImageReadMode.RGB)

        if self.__preprocess:
            image = self.__transform(image)
        if self.__split == 'test':
            return image

        label: Tensor = torch.tensor(self.__data.iloc[index].loc['Class'])
        return image, label

    def __len__(self) -> int:
        return len(self.__data)


def load_data(path: pl.Path, disjoint: bool = True, preprocess: bool = True) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]] | t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
    # Read all subsets
    train_data = GenImageDataset(path, 'train', preprocess)
    valid_data = GenImageDataset(path, 'val', preprocess)
    test_data = GenImageDataset(path, 'test', preprocess)

    # Read all subsets
    test_data = t.cast(data.Dataset[Tensor], test_data)
    valid_data = t.cast(data.Dataset[t.Tuple[Tensor, Tensor]], valid_data)
    train_data = t.cast(data.Dataset[t.Tuple[Tensor, Tensor]], train_data)

    # Merge validation with training data
    if not disjoint:
        train_data = data.ConcatDataset([train_data, valid_data])
        return train_data, test_data

    # Otherwise keep all of them separate
    return train_data, valid_data, test_data


def load_batched_data(path: pl.Path, split: TrainSplit, *, seed=7982, **kwargs):
    # Load disjoint subsets
    data = load_data(path, True, True)
    train_ds = t.cast(Dataset[Tuple[Tensor, Tensor]], data[0])
    valid_ds = t.cast(Dataset[Tuple[Tensor, Tensor]], data[1])
    test_ds = t.cast(Dataset[Tensor], t.cast(t.Any, data)[2])

    # Allow reproducibility
    torch.manual_seed(seed)
    gen = torch.Generator('cpu').manual_seed(seed)

    # Loading settings across all subsets
    settings = {
        'prefetch_factor': kwargs.pop('prefetch_factor', None),
        'batch_size': kwargs.pop('batch_size', None),
        'num_workers': kwargs.pop('num_workers', 0),
        'pin_memory': pin_memory,
    }

    # Decide how to split the training data
    if isinstance(split, tuple):
        train_ds = ConcatDataset([train_ds, valid_ds])
        train_ds, valid_ds = random_split(train_ds, split, gen)
    elif split == 'train':
        train_ds = ConcatDataset([train_ds, valid_ds])
        valid_ds = None
    elif split == 'disjoint':
        pass

    # Create dataloaders for each subset
    train_dl = DataLoader(train_ds, shuffle=True, **settings, generator=gen)
    valid_dl = DataLoader(valid_ds, shuffle=True, **settings, generator=gen) if valid_ds else None
    test_dl = DataLoader(test_ds, shuffle=False, **settings, generator=gen)
    return train_dl, valid_dl, test_dl

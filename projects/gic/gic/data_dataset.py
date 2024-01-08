import torch
from torch import Tensor
from torch import tensor
import torchvision as tn
import torchvision.transforms.v2.functional as F
import typing as t
from typing import TypeAlias
import pathlib as pl
import pandas as ps
import torch.utils.data as data


DataSplit: TypeAlias = t.Literal['train', 'valid', 'test']


class GICDataset(data.Dataset[t.Tuple[Tensor, Tensor] | Tensor]):
    # Precomputed dataset statistics (train + validation)
    mean: Tensor = tensor([0.4282, 0.4064, 0.3406])
    std: Tensor = tensor([0.2144, 0.2187, 0.2046])
    num_classes: int = 100

    def __init__(self,
                 path: pl.Path,
                 split: DataSplit) -> None:
        super(GICDataset, self).__init__()

        # Map names
        if split == 'valid':
            self.split_: str = 'val'
        else:
            self.split_: str = split

        # Paths
        self.__root = path
        self.__meta = self.__root / f'{self.split_}.csv'
        self.__imag = self.__root / f'{self.split_}_images'

        # Internal data table
        self.data_ = ps.read_csv(self.__meta)

    def __getitem__(self, index: int):
        image_name: str = self.data_.iloc[index].loc['Image']
        image_path: str = str(self.__imag / image_name)
        image: Tensor = tn.io.read_image(image_path, tn.io.ImageReadMode.RGB)
        image = F.to_dtype(image, torch.float, scale=True)

        if self.split_ == 'test':
            return image

        label: Tensor = torch.tensor(self.data_.iloc[index].loc['Class'])
        return (image, label)

    def __len__(self) -> int:
        return len(self.data_)

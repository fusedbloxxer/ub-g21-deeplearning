import torch
from torch import Tensor
import torchvision as tn
from torchvision.transforms.v2 import Compose, ToDtype, Resize, Normalize
import typing as t
import pathlib as pl
import pandas as ps
import torch.utils.data as data


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


def load_data(path: pl.Path, disjoint: bool=True, preprocess: bool=True) -> t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]] | t.Tuple[data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[t.Tuple[Tensor, Tensor]], data.Dataset[Tensor]]:
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

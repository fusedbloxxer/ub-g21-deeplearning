import torch
from torch import Tensor
import torchvision as tn
from torchvision.transforms.v2 import Compose, ToDtype, Resize, Normalize
import typing as t
import pathlib as pl
import pandas as ps
import torch.utils.data as data


DataSplit: t.TypeAlias = t.Literal['train', 'val', 'test']
DatasetXY: t.TypeAlias = data.Dataset[t.Union[t.Tuple[Tensor, Tensor], Tensor]]


class GenImageDataset(DatasetXY):
    def __init__(self, path: pl.Path, split: DataSplit, preprocess: bool = True) -> None:
        super(GenImageDataset, self).__init__()
        self.__split = split

        self.__root = path
        self.__meta = self.__root / f'{self.__split}.csv'
        self.__imag = self.__root / f'{self.__split}_images'

        self.data_ = ps.read_csv(self.__meta)

        self.__preprocess = preprocess
        self.__transform = Compose([
            ToDtype(dtype=torch.float32, scale=True),
            Resize(size=(224, 244), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor] | Tensor:
        image_name: str = self.data_.iloc[index].loc['Image']
        image_path: str = str(self.__imag / image_name)
        image: Tensor = tn.io.read_image(image_path, tn.io.ImageReadMode.RGB)

        if self.__preprocess:
            image = self.__transform(image)

        if self.__split == 'test':
            return image

        label: Tensor = torch.tensor(self.data_.iloc[index].loc['Class'])
        return image, label

    def __len__(self) -> int:
        return len(self.data_)


class AugementedDataset(DatasetXY):
    def __init__(self, dataset: DatasetXY) -> None:
        super().__init__()
        self.dataset_ = dataset

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor] | Tensor:
        return self.dataset_.__getitem__(index)

    def __len__(self) -> int:
        return len(t.cast(t.Sequence, self.dataset_))

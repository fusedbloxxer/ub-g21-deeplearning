from typing import Tuple, cast
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import lightning as tl
import pathlib as pl

from . import num_workers, prefetch_factor, pin_memory, gen_torch
from .data_utils import TrainSplit, load_batched_data


class GICDataModule(tl.LightningDataModule):
    def __init__(self,
                 path: pl.Path,
                 batch_size: int,
                 split: TrainSplit=(0.75, 0.25),
                 pin: bool = pin_memory,
                 workers: int = num_workers,
                 prefetch: int | None = prefetch_factor,
                 gen: torch.Generator=gen_torch,
                 ) -> None:
        super().__init__()

        # Loading and reproducibility settings
        self.__path = path
        self.__gen = gen

        # Training settings
        self.batch_sz = batch_size
        self.__split: TrainSplit = split

        # Hardware specific settings
        self.__pin = pin
        self.__workers = workers
        self.__prefetch = prefetch

    def setup(self, stage: str):
        # Load all subsets of data
        loaders = load_batched_data(self.__path, self.__split, self.__gen, batch_size=self.batch_sz, num_workers=self.__workers, prefetch_factor=self.__prefetch, pin_memory=self.__pin)
        self.train_dl, self.valid_dl, self.test_dl = loaders

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return cast(DataLoader[Tuple[Tensor, Tensor]], self.train_dl)

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return cast(DataLoader[Tuple[Tensor, Tensor]], self.valid_dl)

    def test_dataloader(self) -> DataLoader[Tensor]:
        return cast(DataLoader[Tensor], self.test_dl)

    def predict_dataloader(self) -> DataLoader[Tensor]:
        return cast(DataLoader[Tensor], self.test_dl)

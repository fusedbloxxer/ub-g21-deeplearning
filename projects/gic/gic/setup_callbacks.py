import typing as t
from typing import cast
import torch
from torch import Tensor, inference_mode
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid as grid
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as pt

from .data_transform import MaskingNoiseTransform


class ReconstructVizCallback(Callback):
    def __init__(self) -> None:
        super(ReconstructVizCallback, self).__init__()
        self.masking_noise = MaskingNoiseTransform()

    def on_train_epoch_end(self, tr: Trainer, model: LightningModule) -> None:
        # Fetch dataset
        dl: DataLoader[Tensor] = cast(DataLoader[Tensor], tr.train_dataloader)
        ds: Dataset[Tensor] = cast(Dataset[Tensor], dl.dataset)

        # Sample data
        X_sample: Tensor = torch.stack([ds[i][0] for i in range(4)], dim=0)
        X_noisy:  Tensor = self.masking_noise(X_sample)
        X_clear:  Tensor = X_sample

        # Denoise using model
        with inference_mode():
            mode: bool = model.training
            model.train(False)
            X_denoised: Tensor = model.forward(X_noisy.to(model.device)).cpu()
            X_reconstr: Tensor = model.forward(X_clear.to(model.device)).cpu()
            model.train(mode)

        # Aggregate all data into grids
        X_clear_grid = grid(X_clear, nrow=2, padding=5).clip(0, 1)
        X_noisy_grid = grid(X_noisy, nrow=2, padding=5).clip(0, 1)
        X_denoised_grid = grid(X_denoised, nrow=2, padding=5).clip(0, 1)
        X_reconstr_grid = grid(X_reconstr, nrow=2, padding=5).clip(0, 1)

        f, ax = pt.subplots(nrows=1, ncols=4, figsize=(20, 20))
        print('Loss: {}'.format(model._metric_train_loss.compute().item()))
        ax[0].imshow(X_clear_grid.permute((1, 2, 0)))
        ax[0].set_xlabel('Original Image')
        ax[1].imshow(X_noisy_grid.permute((1, 2, 0)))
        ax[1].set_xlabel('Masked Image')
        ax[2].imshow(X_reconstr_grid.permute((1, 2, 0)))
        ax[2].set_xlabel('Reconstructed Image')
        ax[3].imshow(X_denoised_grid.permute((1, 2, 0)))
        ax[3].set_xlabel('Denoised Image')
        pt.show()
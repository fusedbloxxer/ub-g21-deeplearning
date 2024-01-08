from typing import Iterable
from torch import no_grad, device
from torch import Tensor, tensor, zeros, cat
from torcheval.metrics.metric import Metric
from sklearn.metrics import confusion_matrix
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


class ConfusionMatrix(Metric[Tensor]):
    def __init__(self, device: device | None = None) -> None:
        super(ConfusionMatrix, self).__init__(device=device)

        # Store labels across predictions
        self._add_state('y_pred', zeros((0,), device=self.device))
        self._add_state('y_true', zeros((0,), device=self.device))

    @no_grad()
    def update(self, logits: Tensor, labels: Tensor) -> 'ConfusionMatrix':
        self.y_pred = cat([self.y_pred, logits.argmax(dim=-1)])
        self.y_true = cat([self.y_true, labels])
        return self

    @no_grad()
    def merge_state(self, metrics: Iterable['ConfusionMatrix']) -> 'ConfusionMatrix':
        self.y_pred = cat([self.y_pred] + [m.y_pred for m in metrics])
        self.y_true = cat([self.y_true] + [m.y_true for m in metrics])
        return self

    @no_grad()
    def compute(self) -> Tensor:
        return tensor(confusion_matrix(self.y_true.cpu(), self.y_pred.cpu()))


class ReconstructVizCallback(Callback):
    def __init__(self, viz_iter: int) -> None:
        super(ReconstructVizCallback, self).__init__()
        self.masking_noise = MaskingNoiseTransform()
        self.viz_iter = viz_iter

    def on_train_epoch_end(self, tr: Trainer, model: LightningModule) -> None:
        if tr.current_epoch == 0 or tr.current_epoch % self.viz_iter != 0:
            return

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

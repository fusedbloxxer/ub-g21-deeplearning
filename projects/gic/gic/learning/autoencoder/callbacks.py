import typing as t
import torch
import torch.nn as nn
from torch import Tensor, inference_mode
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid as grid
import lightning as tl
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from ...data.transform import tr_perturb, tr_preprocess


class ReconstructVizCallback(Callback):
    def __init__(self) -> None:
        super(ReconstructVizCallback, self).__init__()

    def on_validation_epoch_end(self, tr: Trainer, model: LightningModule) -> None:
        if not tr.val_dataloaders:
            return

        # Fetch data
        lr = t.cast(DataLoader[t.Tuple[Tensor, Tensor]], tr.val_dataloaders)
        ds = lr.dataset
        X_true: Tensor = tr_preprocess(torch.stack([ds[i][0] for i in range(16)], dim=0)).to(model.device)
        X_noisy: Tensor = tr_preprocess(tr_perturb(tr_preprocess.denorm(X_true)))

        # Denoise using model
        with inference_mode():
            mode = model.training
            model.train(False)
            X_pred: Tensor = model.forward(X_noisy)
            model.train(mode)

        # Denorm all data
        X_true = grid(tr_preprocess.denorm(X_true).cpu(), 4, padding=5)
        X_pred = grid(tr_preprocess.denorm(X_pred).cpu(), 4, padding=5)
        X_noisy = grid(tr_preprocess.denorm(X_noisy).cpu(), 4, padding=5)
        t.cast(WandbLogger, tr.logger).log_image('X_true', [X_true])
        t.cast(WandbLogger, tr.logger).log_image('X_noisy', [X_noisy])
        t.cast(WandbLogger, tr.logger).log_image('X_pred', [X_pred])

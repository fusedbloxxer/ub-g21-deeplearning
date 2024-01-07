import typing as t
import torch
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt
from datetime import datetime as dt

from . import DATA_PATH, CKPT_PATH, SUBMISSION_PATH, wn_logger_fn
from .model_densecnn import DenseCNNClassifier
from .model_rescnn import ResCNNClassifier
from .data_dataloader import GICDataModule
from .data_dataset import GICDataset


# Select best model
model = DenseCNNClassifier(
    num_classes=GICDataset.num_classes,
    lr=6e-4,
    inner=4,
    repeat=4,
    features=32,
    augment=True,
    augment_n=1,
    augment_m=11,
    dense=224,
    pool='max',
    activ_fn='SiLU',
    f_drop=0.250,
    c_drop=0.125,
    weight_decay=3e-4,
    factor_c=1,
    factor_t=1,
)

# Separete training & validation subsets for evaluation
start_time = dt.now().strftime(r'%d_%b_%Y_%H:%M')
ckpt_path = CKPT_PATH / model.name
loader = GICDataModule(DATA_PATH, 32, 'disjoint')
logger = wn_logger_fn(
    name=f"{model.name}_Valid_{start_time}",
)
ckpt = ModelCkpt(
    save_top_k=-1,
    every_n_epochs=1,
    dirpath=ckpt_path,
    save_on_train_epoch_end=True,
    filename=f'{model.name}_{{epoch:03d}}')
trainer = Trainer(
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0,
    limit_val_batches=1.0,
    callbacks=[ckpt],
    max_epochs=250,
    logger=logger)
trainer.fit(model, datamodule=loader)

# Perform final validation using last weights
trainer.validate(model, datamodule=loader)

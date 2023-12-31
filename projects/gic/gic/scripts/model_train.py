import typing as t
import torch
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt
from datetime import datetime as dt

from gic import DATA_PATH, PROJECT_NAME, LOG_PATH, CKPT_PATH, SUBMISSION_PATH
from gic.data.dataset import GICDataset
from gic.data.dataloader import GICDataModule
from gic.learning.focalnet.wrappers import FocalNetClassifier


# Select best model
model = FocalNetClassifier(
    lr=4e-4,
    groups=8,
    repeat=3,
    layers=2,
    chan=128,
    dense=224,
    augment_n=1,
    augment_m=11,
    augment=True,
    dropout=0.15,
    reduce="max",
    weight_decay=8e-3,
    conv_order="2 1 0",
    dropout_dense=0.30,
    norm_layer='batch',
    drop_type='spatial',
    activ_fn='LeakyReLU',
    num_classes=GICDataset.num_classes,
)

# Join training & validation subsets for final submission
start_time = dt.now().strftime(r'%d_%b_%Y_%H:%M')
ckpt_path = CKPT_PATH / model.name
loader = GICDataModule(DATA_PATH, 32, 'joint')
logger = WandbLogger(
    name=f"{model.name}_Train_{start_time}",
    project=PROJECT_NAME,
    save_dir=LOG_PATH)
ckpt = ModelCkpt(
    save_top_k=-1,
    every_n_epochs=1,
    dirpath=ckpt_path,
    save_on_train_epoch_end=True,
    filename=f'{model.name}_{{epoch:03d}}')
trainer = Trainer(
    enable_checkpointing=True,
    check_val_every_n_epoch=0,
    num_sanity_val_steps=0,
    limit_val_batches=0,
    callbacks=[ckpt],
    max_epochs=3, # 288
    logger=logger)
trainer.fit(model, datamodule=loader)

# Load trained weights from disk
model = FocalNetClassifier.load_from_checkpoint(ckpt_path / f'{model.name}_epoch=002.ckpt')

# Predict on test data
y_hat = t.cast(t.List[Tensor], trainer.predict(model, datamodule=loader, return_predictions=True))
preds = torch.cat(y_hat, dim=0)

# Create submission file
data = GICDataset(DATA_PATH, 'test').data_
data['Class'] = preds
data.to_csv(SUBMISSION_PATH, index=False)

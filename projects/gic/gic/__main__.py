import sys as s
import torch
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt

from gic import *
from gic.data.dataset import GICDataset
from gic.data.dataloader import GICDataModule
from gic.learning.focalnet.objectives import FocalNetObjective
from gic.learning.focalnet.wrappers import FocalNetClassifier


# # Find hyperparams that maximize the validation f1 score
# sampler = opt.samplers.TPESampler(n_startup_trials=10)
# s = opt.create_study(direction='maximize', sampler=sampler)
# s.optimize(FocalNetObjective(), n_trials=250, show_progress_bar=True)

# Select best model
model = FocalNetClassifier(
    lr=4e-4,
    groups=8,
    norm_layer='batch',
    drop_type='spatial',
    num_classes=GICDataset.num_classes,
    augment=True,
    augment_n=1,
    augment_m=11,
    repeat=3,
    dropout=0.15,
    layers=2,
    chan=128,
    weight_decay=8e-3,
    dense=224,
    reduce="max",
    dropout_dense=0.30,
    activ_fn='LeakyReLU',
    conv_order="2 1 0"
)

# Join training & validation subsets for final submission
loader = GICDataModule(DATA_PATH, 32, 'joint')
logger = WandbLogger(
    name="FocalNet - Submission - 27/12/'23-11:21",
    project=PROJECT_NAME,
    save_dir=LOG_PATH,
)
ckpt = ModelCkpt(
    save_top_k=-1,
    every_n_epochs=1,
    dirpath=CKPT_PATH,
    save_on_train_epoch_end=True,
    filename='focalnet_{epoch:03d}_{train_f1_score:.3f}',
)
trainer = Trainer(
    default_root_dir=CKPT_PATH,
    enable_checkpointing=True,
    check_val_every_n_epoch=0,
    num_sanity_val_steps=0,
    limit_val_batches=0,
    callbacks=[ckpt],
    max_epochs=250,
    logger=logger,
)
# trainer.fit(model, datamodule=loader)

model = FocalNetClassifier.load_from_checkpoint(CKPT_PATH / 'focalnet_epoch=228_train_f1_score=0.787.ckpt', name='FocalNet')

# Predict on test data
y_hat = t.cast(t.List[Tensor], trainer.predict(model, datamodule=loader, return_predictions=True))
preds = torch.cat(y_hat, dim=0)

# Create submission file
test = GICDataset(DATA_PATH, 'test')
data = test.data_
data['Class'] = preds
data.to_csv(SUBMISSION_PATH, index=False)

# 236, 180, 189, 208
# 228 - good!
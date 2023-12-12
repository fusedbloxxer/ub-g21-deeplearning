import os as so
import sys as s
import pathlib as pl
from torch import Tensor
from functools import partial
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from gic import *
from gic.data.dataset import GICDataset
from gic.data.dataloader import GICDataLoader
from gic.learning.focalnet.factory import factory
from gic.learning.focalnet.wrappers import FocalNetClassifier
from gic.learning.objectives import search


# Perform HyperParam Search
hparam_search = partial(search, factory=factory)
pruner = opt.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=40)
s = opt.create_study(direction='maximize', pruner=pruner)
s.optimize(hparam_search, n_trials=150, show_progress_bar=True)

# # Select best model
# model = FocalNetClassifier(
#     activ_fn='LeakyReLU',
#     chan=56,
#     conv_order='0 1 2',
#     dense=256,
#     drop_type='spatial',
#     dropout=0.3,
#     dropout_dense=0.4965179817068612,
#     groups=8,
#     lr=0.0007285659930586672,
#     norm_layer='batch',
#     num_classes=GICDataset.num_classes,
#     reduce='max',
#     repeat=2,
#     weight_decay=0.004524643600968936
# )

# # Join training & validation subsets for final submission
# loader = GICDataLoader(DATA_PATH, 32, True)
# logger = WandbLogger(project=PROJECT_NAME, name='FocalNet - Valid', save_dir=LOG_PATH)
# trainer = Trainer(max_epochs=130, enable_checkpointing=False, logger=logger, num_sanity_val_steps=0)
# trainer.fit(model, datamodule=loader)

# # Predict on test data
# y_hat: t.List[Tensor] = t.cast(t.List[Tensor], trainer.predict(model, datamodule=loader, return_predictions=True))
# preds = torch.cat(y_hat, dim=0)

# # Create submission file
# test = GICDataset(DATA_PATH, 'test')
# data = test.data_
# data['Class'] = preds
# data.to_csv(SUBMISSION_PATH, index=False)

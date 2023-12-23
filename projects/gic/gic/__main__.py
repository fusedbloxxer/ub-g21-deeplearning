import sys as s
import torch
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

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
    augment_m=9,
    repeat=3,
    dropout=0.12,
    layers=2,
    chan=128,
    weight_decay=0.0016,
    dense=256,
    reduce="max",
    dropout_dense=0.30,
    activ_fn='LeakyReLU',
    conv_order="2 1 0"
)

# Join training & validation subsets for final submission
loader = GICDataModule(DATA_PATH, 32, 'joint')
logger = WandbLogger(project=PROJECT_NAME, name='FocalNet - Valid', save_dir=LOG_PATH)
trainer = Trainer(max_epochs=156, enable_checkpointing=False, num_sanity_val_steps=0, limit_val_batches=0, logger=logger)
trainer.fit(model, datamodule=loader)
trainer.validate(model, datamodule=loader)

# Predict on test data
y_hat: t.List[Tensor] = t.cast(t.List[Tensor], trainer.predict(model, datamodule=loader, return_predictions=True))
preds = torch.cat(y_hat, dim=0)

# Create submission file
test = GICDataset(DATA_PATH, 'test')
data = test.data_
data['Class'] = preds
data.to_csv(SUBMISSION_PATH, index=False)

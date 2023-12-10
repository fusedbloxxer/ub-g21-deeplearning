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
from gic.learning.densecnn.wrappers import DenseCNNClassifierModule
from gic.learning.objectives import search


# Perform HyperParam Search
hparam_search = partial(search, factory=factory)
pruner = opt.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=40)
s = opt.create_study(direction='maximize', pruner=pruner)
s.optimize(hparam_search, n_trials=100, show_progress_bar=True)

# Select best model
model = DenseCNNClassifierModule(
    activ_fn='GELU',
    c_drop=0.2,
    dense=512,
    f_drop=0.3,
    factor_c=1,
    factor_t=1,
    features=12,
    inner=3,
    num_classes=GICDataset.num_classes,
    pool='avg',
    repeat=3,
    lr=0.0005391117462806324,
    weight_decay=0.00005082016028675467,
)

# Join training & validation subsets for final submission
loader = GICDataLoader(DATA_PATH, 32, True, 'joint')
logger = WandbLogger(project=PROJECT_NAME, name='DenseCNN - Valid Train', save_dir=LOG_PATH)
trainer = Trainer(max_epochs=100, enable_checkpointing=False, logger=logger)
trainer.fit(model, datamodule=loader)

# Predict on test data
y_hat: t.List[Tensor] = t.cast(t.List[Tensor], trainer.predict(model, datamodule=loader, return_predictions=True))
preds = torch.cat(y_hat, dim=0)

# Create submission file
test = GICDataset(DATA_PATH, 'test')
data = test.data_
data['Class'] = preds
data.to_csv(SUBMISSION_PATH, index=False)

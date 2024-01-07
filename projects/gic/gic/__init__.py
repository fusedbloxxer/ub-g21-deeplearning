import torch
import os as so
import numpy as ny
import random as rng
import pathlib as pl
import optuna as opt
import typing as t
import lightning as tl
from lightning.pytorch.loggers import WandbLogger
from functools import partial

# Development
IS_DEBUG = False
IS_RELEASE = False
IS_PROFILING = False

# Paths
ROOT_PATH = pl.Path('.')
LOG_PATH = ROOT_PATH / 'log'
CKPT_PATH = ROOT_PATH / 'ckpt'
DATA_PATH = ROOT_PATH / 'data'
GIT_PATH = ROOT_PATH / '..' / '..'
NOTEBOOKS_PATH = ROOT_PATH / 'notebooks'
SUBMISSIONS_PATH = ROOT_PATH / 'submissions'

# Versioning
PROJECT_NAME = 'Generated Image Classification'
SUBMISSION_NAME = 'submission_49'
SUBMISSION_PATH = SUBMISSIONS_PATH / f'{SUBMISSION_NAME}.csv'
so.environ['WANDB_NOTEBOOK_NAME'] = str(NOTEBOOKS_PATH / 'main.ipynb')
wn_logger_fn = partial(WandbLogger, dir=LOG_PATH, offline=IS_RELEASE, anonymous=IS_RELEASE, project=PROJECT_NAME)

# Hardware
mix_float = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefetch_factor = 100
pin_memory= True
num_workers= 4

# Reproducibility
SEED = 10_056
rng.seed(SEED)
ny.random.seed(SEED)
torch.manual_seed(SEED)
tl.seed_everything(SEED)
gen_numpy = ny.random.default_rng(SEED)
gen_torch = torch.Generator('cpu').manual_seed(SEED)

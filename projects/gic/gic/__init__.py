import torch
import os as so
import numpy as ny
import random as rng
import pathlib as pl
import optuna as opt
import typing as t
from optuna.integration import WeightsAndBiasesCallback
from git import Repo
import lightning as tl

# Development
debug = False
release = False
profiling = False

# Paths
ROOT_PATH = pl.Path('..')
GIT_PATH = ROOT_PATH / '..' / '..'
DATA_PATH = ROOT_PATH / 'data'
CKPT_PATH = ROOT_PATH / 'ckpt'
LOG_PATH = ROOT_PATH / 'log'
NOTEBOOKS_PATH = ROOT_PATH / 'notebooks'
SUBMISSIONS_PATH = ROOT_PATH / 'submissions'

# Versioning
git_repo = Repo(GIT_PATH)
SUBMISSION_PATH = SUBMISSIONS_PATH / f'{git_repo.active_branch.name}.csv'
so.environ['WANDB_NOTEBOOK_NAME'] = str(NOTEBOOKS_PATH / 'main.ipynb')

# Hardware
mix_float = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefetch_factor = None
pin_memory= False
num_workers= 0

# Reproducibility
SEED = 7982
rng.seed(SEED)
ny.random.seed(SEED)
torch.manual_seed(SEED)
tl.seed_everything(SEED)
gen_numpy = ny.random.default_rng(SEED)
gen_torch = torch.Generator('cpu').manual_seed(SEED)

# Constants
CONST_NUM_CLASS = 100

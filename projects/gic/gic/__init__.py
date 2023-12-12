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
git_repo = Repo(GIT_PATH)
PROJECT_NAME = 'Generated Image Classification'
SUBMISSION_NAME = git_repo.active_branch.name
SUBMISSION_PATH = SUBMISSIONS_PATH / f'{git_repo.active_branch.name}.csv'
so.environ['WANDB_NOTEBOOK_NAME'] = str(NOTEBOOKS_PATH / 'main.ipynb')

# Hardware
mix_float = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefetch_factor = 1_000
pin_memory= True
num_workers= 4

# Reproducibility
SEED = 8108
rng.seed(SEED)
ny.random.seed(SEED)
torch.manual_seed(SEED)
tl.seed_everything(SEED)
gen_numpy = ny.random.default_rng(SEED)
gen_torch = torch.Generator('cpu').manual_seed(SEED)

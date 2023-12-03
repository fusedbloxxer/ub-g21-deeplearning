import torch
import os as so
import numpy as ny
import random as rng
import pathlib as pl
import optuna as opt
import typing as t
import wandb as wn
from optuna.integration import WeightsAndBiasesCallback
from git import Repo

# Development
debug = True
release = False

# Paths
ROOT_PATH = pl.Path('..')
GIT_PATH = ROOT_PATH / '..' / '..'
DATA_PATH = ROOT_PATH / 'data'
CKPT_PATH = DATA_PATH / 'ckpt'
SUBMISSIONS_PATH = ROOT_PATH / 'submissions'
NOTEBOOKS_PATH = ROOT_PATH / 'notebooks'
LOG_PATH = ROOT_PATH / 'log'

# Versioning
git_repo = Repo(GIT_PATH)
SUBMISSION_PATH = SUBMISSIONS_PATH / f'{git_repo.active_branch.name}.csv'

# Tracking
so.environ['WANDB_NOTEBOOK_NAME'] = str(NOTEBOOKS_PATH / 'main.ipynb')
login_settings = {}
init_settings = {
    'project': 'Generated Image Classification',
    'notes': 'Experimentation',
    'tags': [git_repo.active_branch.name, 'DenseCNN', 'Augment']
}

# Use an account only during development
if not release:
    so.environ['WANDB_MODE'] = 'online'
    login_settings.update({
        'anonymous': 'never',
    })
else:
    so.environ['WANDB_MODE'] = 'offline'
    login_settings.update({
        'anonymous': 'must',
    })
    init_settings.update({
        'mode': 'disabled',
    })

# Init Tracking
wn.login(**login_settings)

# Integrate Tracking with Optuna
wn_callback = WeightsAndBiasesCallback(
    wandb_kwargs=init_settings,
    metric_name='f1_score',
    as_multirun=True,
)

# Hardware
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefetch_factor = 1_000
num_workers = 4
pin_memory = True
mix_float = True
profiling = False

# Reproducibility
SEED = 7982
rng.seed(SEED)
ny.random.seed(SEED)
torch.manual_seed(SEED)
gen_numpy = ny.random.default_rng(SEED)
gen_torch = torch.Generator('cpu').manual_seed(SEED)

# Constants
CONST_NUM_CLASS = 100
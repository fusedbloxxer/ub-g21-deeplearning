import typing as t
from typing import cast, Any, Literal
import argparse as ap
from argparse import ArgumentDefaultsHelpFormatter as DefaultFormatter
from argparse import Namespace, ArgumentParser as Parser
from pathlib import Path
import torch
import numpy as ny
import random as rng
import lightning as tl
from functools import partial
from lightning.pytorch.loggers import WandbLogger as Logger
from dataclasses import dataclass


@dataclass
class Config(object):
    device: torch.device
    gen_torch: torch.Generator
    gen_numpy: ny.random.Generator
    logger_factory: partial[Logger]
    prefetch: int
    workers: int
    pin: bool
    data_path: Path
    ckpt_path: Path
    subm_path: Path
    log_path: Path
    args: Namespace
    batch_size: int
    epochs: int
    project_name: str
    model: Literal['rescnn', 'densecnn', 'dae']
    command: t.Literal['train', 'valid', 'ensebmle', 'optimize']


def make_parser():
    root_command = ap.ArgumentParser(prog='gic', formatter_class=DefaultFormatter)
    root_command.add_argument('--sub-name', type=str, default='submission.csv', help='the name of the submission file')

    # Configure IO paths
    gr_path = root_command.add_argument_group('path')
    gr_path.add_argument('--ckp-path', type=Path, default=Path('.', 'ckpt'), help='where model weights will be saved')
    gr_path.add_argument('--img-path', type=Path, default=Path('.', 'data'), help='where the data will be loaded from')
    gr_path.add_argument('--sub-path', type=Path, default=Path('.', 'submissions'), help='where submissions will be saved')
    gr_path.add_argument('--log-path', type=Path, default=Path('.', 'log'), help='where the temporary logs will be saved at')

    # Configure runtime
    gr_runtime = root_command.add_argument_group('runtime')
    gr_runtime.add_argument('--debug', action='store_false', help='disable cloud logging with Wandb')
    gr_runtime.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='device used for training and inference')
    gr_runtime.add_argument('-s', '--seed', action='store', type=int, default=10_056, help='reproducibility')
    gr_runtime.add_argument('--prefetch', type=int, default=4, help='preload batches')
    gr_runtime.add_argument('--pinning', type=bool, default=True, help='enable memory pinning')
    gr_runtime.add_argument('--workers', type=int, default=4, help='number of processes used to load batches')

    # Select architecture
    gr_model = root_command.add_argument_group('model', description='select and customize the architecture')
    gr_model.add_argument('model', choices=['rescnn', 'densecnn', 'dae'], help='neural network architecture')
    gr_model.add_argument('--batch-size', type=int, default=32, help='how many images should be in a batch')
    gr_model.add_argument('--epochs', type=int, default=250, help='the number of epochs used')

    # Allow the user to optimize/train/validate/submit
    root_subcommands = root_command.add_subparsers(title='command', dest='command', required=True)
    add_subcommand_train(root_subcommands)
    add_subcommand_valid(root_subcommands)
    add_subcommand_ensemble(root_subcommands)
    add_subcommand_optimize(root_subcommands)
    return root_command


def add_subcommand_train(root_subcommands: ap._SubParsersAction):
    train_parser: Parser = root_subcommands.add_parser(name='train', help='train the model over the whole dataset')


def add_subcommand_valid(root_subcommands: ap._SubParsersAction):
    valid_parser: Parser = root_subcommands.add_parser(name='valid', help='train and validate the model against the validation set')


def add_subcommand_ensemble(root_subcommands: ap._SubParsersAction):
    ensemble_parser: Parser = root_subcommands.add_parser(name='ensemble', help='create a bagging ensemble using the model architecture')
    ensemble_parser.add_argument('--members', type=int, default=5, help='the number of models present in the ensemble')
    ensemble_parser.add_argument('mode', choices=['train', 'valid'], help='train an ensemble of models')


def add_subcommand_optimize(root_subcommands: ap._SubParsersAction):
    optimize_parser: Parser = root_subcommands.add_parser(name='optimize', help='search for the hyperparameters that minimize validation f1_score')
    optimize_parser.add_argument('--trials', type=int, default=250, help='the number of trials per study')


def setup_env(args: Namespace):
    # Configure device settings
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    # Allow results to be reproduced
    rng.seed(args.seed)
    ny.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tl.seed_everything(args.seed)
    gen_numpy = ny.random.default_rng(args.seed)
    gen_torch = torch.Generator('cpu').manual_seed(args.seed)
    project_name = 'Generated Image Classification'

    # Configure project-wide logging
    make_logger = partial(
        Logger,
        dir=args.log_path,
        offline=args.debug,
        anonymous=args.debug,
        project=project_name,
    )

    # Aggregate all configuration files
    return Config(
        args=args,
        device=device,
        pin=args.pinning,
        gen_torch=gen_torch,
        gen_numpy=gen_numpy,
        workers=args.workers,
        prefetch=args.prefetch,
        log_path=args.log_path,
        data_path= args.img_path,
        ckpt_path= args.ckp_path,
        logger_factory=make_logger,
        model=cast(Any, args.model),
        batch_size=args.batch_size,
        epochs=args.epochs,
        command=args.command,
        project_name=project_name,
        subm_path=args.sub_path / args.sub_name,
    )

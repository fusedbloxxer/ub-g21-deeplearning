import typing as t
import argparse as ap
from argparse import ArgumentDefaultsHelpFormatter as DefaultFormatter
from pathlib import Path
import torch
import os as so
import numpy as ny
import random as rng
import pathlib as pl
import optuna as opt
import typing as t
import optuna as opt
from functools import partial
from torch import Tensor
import lightning as tl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt
from datetime import datetime as dt

from gic.model_ensemble import BaggingEnsemble
from gic.data_dataloader import GICDataModule
from gic.data_dataset import GICDataset
from gic.model_densecnn import DenseCNNObjective
from gic.model_rescnn import ResCNNObjective
from gic.model_dae import DAEClasifierObjective
from gic.hyperparams import get_model_hparams
from gic.setup import make_parser, setup_env


# Retrieve CLI args and prepare environment
root_parser = make_parser()
cli_args = root_parser.parse_args()
config = setup_env(cli_args)
print(config)

# Customize dataset
data_module_fn = partial(GICDataModule,
    config.data_path,
    config.batch_size,
    config.pin,
    config.workers,
    config.prefetch,
    config.gen_torch,
)

# Run user given operation
match config.command:
    case 'optimize':
        # Select search strategy
        strategy = opt.samplers.TPESampler(n_startup_trials=10)
        race = opt.create_study(direction='maximize', sampler=strategy)

        # Reshuffle the dataset on each run
        data = lambda: data_module_fn(split=(0.75, 0.25))

        # Select objective
        match config.model:
            case   'rescnn': obj = ResCNNObjective(config.batch_size, config.epochs, data, config.logger_factory)
            case 'densecnn': obj = DenseCNNObjective(config.batch_size, config.epochs, data, config.logger_factory)
            case      'dae': obj = DAEClasifierObjective(config.batch_size, config.epochs, data, config.logger_factory, config.ckpt_path)

        # Perform study
        race.optimize(obj, n_trials=config.args.trials, show_progress_bar=True)
    case 'train':
        # Select the best model architecture
        model_type, model_hparams = get_model_hparams(config.model, config.ckpt_path)
        model = model_type(**model_hparams)

        # Join training & validation subsets for final submission
        start_time = dt.now().strftime(r'%d_%b_%Y_%H:%M')
        ckpt_path = config.ckpt_path / 'train' / model.name
        data = data_module_fn(split='joint')
        logger = config.logger_factory(name=f"{model.name}_Train_{start_time}")
        ckpt = ModelCkpt(
            save_top_k=-1,
            every_n_epochs=1,
            dirpath=ckpt_path,
            save_on_train_epoch_end=True,
            filename=f'{model.name}_{{epoch:03d}}')
        trainer = Trainer(
            enable_checkpointing=True,
            check_val_every_n_epoch=0,
            max_epochs=config.epochs,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[ckpt],
            logger=logger)
        trainer.fit(model, datamodule=data)

        # Load trained weights from disk and predict on test data
        model = model_type.load_from_checkpoint(ckpt_path / f'{model.name}_epoch={config.epochs - 1:03d}.ckpt')
        y_hat = t.cast(t.List[Tensor], trainer.predict(model, datamodule=data, return_predictions=True))
        preds = torch.cat(y_hat, dim=0)

        # Create submission file
        config.args.sub_path.mkdir(parents=True, exist_ok=True)
        data = GICDataset(config.data_path, 'test').data_
        data['Class'] = preds
        data.to_csv(config.subm_path, index=False)
    case 'valid':
        # Select the best model architecture
        model_type, model_hparams = get_model_hparams(config.model, config.ckpt_path)
        model = model_type(**model_hparams)

        # Separete training & validation subsets for evaluation
        start_time = dt.now().strftime(r'%d_%b_%Y_%H:%M')
        ckpt_path = config.ckpt_path / 'valid' / model.name
        data = data_module_fn(split='disjoint')
        logger = config.logger_factory(name=f"{model.name}_Valid_{start_time}")
        ckpt = ModelCkpt(
            save_top_k=-1,
            every_n_epochs=1,
            dirpath=ckpt_path,
            save_on_train_epoch_end=True,
            filename=f'{model.name}_{{epoch:03d}}')
        trainer = Trainer(
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            limit_val_batches=1.0,
            callbacks=[ckpt],
            max_epochs=config.epochs,
            logger=logger)
        trainer.fit(model, datamodule=data)

        # Perform final validation using last weights
        trainer.validate(model, datamodule=data)
    case 'ensemble':
        # Select the best model architecture
        model_type, model_hparams = get_model_hparams(config.model, config.ckpt_path)

        # Create an ensemble of five identic models
        ensemble = BaggingEnsemble(
            model_type=model_type,
            n_models=config.args.members,
            args=t.cast(t.Any, model_hparams),
            ckpt_path=config.ckpt_path,
            log_path=config.log_path,
            device=config.device,
            project_name=config.project_name,
            logger_fn=config.logger_factory,
        )

        # Train and Infer/Validate
        match config.args.mode:
            case 'train':
                # Use train and validation subsets for final submission
                data = data_module_fn(split='joint')

                # Train the ensemble in sequential manner
                ensemble.fit(config.epochs, data, validate=False)

                # Load the final epoch for subission
                ensemble.load_ensemble(config.epochs - 1)

                # Predict over test data
                preds = ensemble.predict(data, 'test').sum(dim=1).argmax(dim=-1)

                # Create submission file
                config.args.sub_path.mkdir(parents=True, exist_ok=True)
                data = GICDataset(config.data_path, 'test').data_
                data['Class'] = preds
                data.to_csv(config.subm_path, index=False)
            case 'valid':
                # Separate train and validation to measure model performance
                data = data_module_fn(split='disjoint')

                # Train the ensemble in sequential manner
                ensemble.fit(config.epochs, data, validate=True)

                # Validate the ensemble
                ensemble.validate(config.epochs - 1, data)
            case _:
                raise ValueError('invalid ensemble --mode {}'.format(config.args.mode))

import typing as t
import torch
import optuna as opt
import typing as t
import optuna as opt
from functools import partial
from torch import Tensor
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCkpt
from lightning.pytorch.callbacks import RichModelSummary as SummaryCallback
from lightning.pytorch.callbacks import RichProgressBar as ProgressCallback
from datetime import datetime as dt
from sklearn.cluster import KMeans
import matplotlib.pyplot as pt
from PIL import Image
import seaborn as sea

from gic.callbacks import ReconstructVizCallback, ConfusionMatrix
from gic.model_ensemble import BaggingEnsemble
from gic.data_dataloader import GICDataModule
from gic.data_dataset import GICDataset
from gic.model_densecnn import DenseCNNObjective
from gic.model_rescnn import ResCNNObjective
from gic.model_dae import DAEClasifierObjective
from gic.model_dae import DAEModule
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

        # Show how the well the model performed
        color_range = sea.color_palette('magma', as_cmap=True)
        f = pt.figure(figsize=(10, 10))
        graph = sea.heatmap(model._metric_valid_confm.compute(), cbar=True, square=True, cmap=color_range, xticklabels=2, yticklabels=2)
        graph.set_ylabel('Ground-Truth Label')
        graph.set_xlabel('Predicted Label')
        graph.set_title('Confusion Matrix')
        pt.show()
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
                valid = torch.tensor(GICDataset(config.data_path, 'valid').data_['Class'].values)

                # Train the ensemble in sequential manner
                ensemble.fit(config.epochs, data, validate=True)

                # Validate the ensemble
                ensemble.validate(config.epochs, data)

                # Showcase the model performance
                preds = ensemble.predict(data, 'valid').sum(dim=1)

                # Show how the well the model performed
                mat = ConfusionMatrix().update(preds, valid).compute()
                color_range = sea.color_palette('magma', as_cmap=True)
                f = pt.figure(figsize=(10, 10))
                graph = sea.heatmap(mat, cbar=True, square=True, cmap=color_range, xticklabels=2, yticklabels=2)
                graph.set_ylabel('Ground-Truth Label')
                graph.set_xlabel('Predicted Label')
                graph.set_title('Confusion Matrix')
                pt.show()
            case _:
                raise ValueError('invalid ensemble --mode {}'.format(config.args.mode))
    case 'denoising':
        # Enforce arg constraint
        if config.model != 'dae':
            raise ValueError('dae model needs to be specified for denoising task')

        # Create AutoEncoder for denoising task
        model = DAEModule(chan=64, latent=64)

        # Create Visualizers to evaluate progress
        prog_clbk = ProgressCallback()
        sum_clbk  = SummaryCallback(max_depth=5)
        viz_clbk  = ReconstructVizCallback(config.args.viz_iter)

        # Load training subset for denoising task
        datamodule = data_module_fn(split='disjoint')

        # Train the DAE
        trainer = Trainer(max_epochs=config.epochs, callbacks=[viz_clbk, prog_clbk, sum_clbk])
        trainer.fit(model, datamodule)

        # Save the weights and load them later for classification
        (config.ckpt_path / 'train' / model.name).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.ckpt_path / 'train' / model.name / f'{model.name}.pt')
    case 'cluster':
        # Enforce arg constraint
        if config.model != 'dae':
            raise ValueError('dae model needs to be specified for cluster task')

        # Create AutoEncoder for denoising task
        model = DAEModule(chan=64, latent=64).to(config.device)
        model.load_state_dict(torch.load(config.ckpt_path / 'train' / model.name / f'{model.name}.pt'))
        model.requires_grad_(False)
        model.eval()

        # Load training data
        data_train = GICDataset(config.data_path, 'train')
        embeddings: t.List[torch.Tensor] = []

        # Perform Img2Embedding
        for i in range(len(data_train)):
            X, _ = data_train[i]
            l = model.autoencoder.encode(X.unsqueeze(0).to(config.device)).cpu()
            embeddings.append(l.flatten())
        train_embeddings = torch.stack(embeddings, dim=0)
        kmeans = KMeans(n_clusters=100)
        kmeans.fit(train_embeddings)

        # Retain the ImagePaths for each image inside each cluster
        groups = {}
        for i, l in zip(range(len(data_train)), kmeans.labels_):
            data_train.data_.iloc[i]
            if l not in groups.keys():
                groups[l] = []
            groups[l].append(data_train.data_.iloc[i]['Image'])

        # Display 16 images inside each cluster
        for i, (g, imgs) in enumerate(sorted(groups.items(), key=lambda x: x[0])):
            f, ax = pt.subplots(1, 16, figsize=(20, 2.5))
            for j in range(min(16, len(imgs))):
                with Image.open(config.data_path / 'train_images' / imgs[j]) as img:
                    ax[j].set_xlabel(f'group: {g}')
                    ax[j].imshow(img)
            pt.show()

from .ensemble_model import ResCNNEnsemble, DenseCNNEnsemble
from .data_dataloader import GICDataModule
from .data_dataset import GICDataset
from . import DATA_PATH


# Separate train and validation to measure model performance
data = GICDataModule(DATA_PATH, 32, 'disjoint')

# Create an ensemble of five models using the best architecture
ensemble = ResCNNEnsemble(
    n_models=5,
    args={
        'lr': 4e-4,
        'chan': 128,
        'groups': 8,
        'repeat': 3,
        'layers': 2,
        'dense': 224,
        'augment_n': 1,
        'augment': True,
        'dropout': 0.15,
        'augment_m': 11,
        'reduce': "max",
        'weight_decay': 8e-3,
        'norm_layer': 'batch',
        'dropout_dense': 0.30,
        'conv_order': "2 1 0",
        'drop_type': 'spatial',
        'activ_fn': 'LeakyReLU',
        'num_classes': GICDataset.num_classes,
    }
)

# Train the ensemble in sequential manner
ensemble.fit(251, data, validate=True)

# Validate the ensemble
ensemble.validate(250, data)

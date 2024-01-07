from . import DATA_PATH, SUBMISSION_PATH
from .ensemble_model import ResCNNEnsemble, DenseCNNEnsemble
from .data_dataloader import GICDataModule
from .data_dataset import GICDataset

# Use train and validation subsets for final submission
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
ensemble.fit(176, data, validate=False)

# Load the final epoch for subission
ensemble.load_ensemble(88)

# Predict over test data
preds = ensemble.predict(data, 'test').sum(dim=1).argmax(dim=-1)

# Create submission file
data = GICDataset(DATA_PATH, 'test').data_
data['Class'] = preds
data.to_csv(SUBMISSION_PATH, index=False)

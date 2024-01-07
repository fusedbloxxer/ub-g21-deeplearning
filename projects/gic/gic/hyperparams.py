from typing import Literal, Tuple
from pathlib import Path
from gic.model_base import ClassifierModule
from gic.model_densecnn import DenseCNNClassifier
from gic.model_rescnn import ResCNNClassifier
from gic.model_dae import DAEClasifier
from gic.data_dataset import GICDataset


def get_model_hparams(model: Literal['rescnn', 'densecnn', 'dae'], ckpt_path: Path) -> Tuple[type[ClassifierModule], dict]:
    # Select the best model architecture
    match model:
        case 'rescnn':
            model_type = ResCNNClassifier
            model_hparams = dict(
                lr=4e-4,
                chan=128,
                groups=8,
                repeat=3,
                layers=2,
                dense=224,
                augment_n=1,
                augment=True,
                dropout=0.15,
                augment_m=11,
                reduce="max",
                weight_decay=8e-3,
                norm_layer="batch",
                dropout_dense=0.30,
                conv_order="2 1 0",
                drop_type="spatial",
                activ_fn="LeakyReLU",
                num_classes= GICDataset.num_classes,
            )
        case 'densecnn':
            model_type = DenseCNNClassifier
            model_hparams = dict(
                lr=6e-4,
                inner=4,
                repeat=4,
                features=32,
                augment=True,
                augment_n=1,
                augment_m=11,
                dense=224,
                pool='max',
                activ_fn='SiLU',
                f_drop=0.250,
                c_drop=0.125,
                weight_decay=3e-4,
                factor_c=1,
                factor_t=1,
                num_classes=GICDataset.num_classes,
            )
        case 'dae':
            model_type = DAEClasifier
            model_hparams = dict(
                lr=6e-4,
                weight_decay=3e-4,
                augment=True,
                augment_n=1,
                augment_m=11,
                dense=224,
                activ_fn='SiLU',
                dropout=0.25,
                batch='norm',
                ckpt_path=ckpt_path,
                num_classes=GICDataset.num_classes,
            )
    return model_type, model_hparams

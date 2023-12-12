from lightning.pytorch import Callback as Wrapper
from optuna.integration import PyTorchLightningPruningCallback as PTLW


class PTLWrapper(PTLW, Wrapper):
    def __init__(self, *args, **kwargs):
        super(PTLWrapper, self).__init__(*args, **kwargs)

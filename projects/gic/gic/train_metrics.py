from typing import Iterable
from torch import no_grad, device
from torch import Tensor, tensor, zeros, cat
from torcheval.metrics.metric import Metric
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(Metric[Tensor]):
    def __init__(self, device: device | None = None) -> None:
        super(ConfusionMatrix, self).__init__(device=device)

        # Store labels across predictions
        self._add_state('y_pred', zeros((0,), device=self.device))
        self._add_state('y_true', zeros((0,), device=self.device))

    @no_grad()
    def update(self, logits: Tensor, labels: Tensor) -> 'ConfusionMatrix':
        self.y_pred = cat([self.y_pred, logits.argmax(dim=-1)])
        self.y_true = cat([self.y_true, labels])
        return self

    @no_grad()
    def merge_state(self, metrics: Iterable['ConfusionMatrix']) -> 'ConfusionMatrix':
        self.y_pred = cat([self.y_pred] + [m.y_pred for m in metrics])
        self.y_true = cat([self.y_true] + [m.y_true for m in metrics])
        return self

    @no_grad()
    def compute(self) -> Tensor:
        return tensor(confusion_matrix(self.y_true.cpu(), self.y_pred.cpu()))

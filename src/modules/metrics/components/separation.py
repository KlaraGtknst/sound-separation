from typing import Any

from torchmetrics import Metric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as sisnr
from torchmetrics.audio import SignalNoiseRatio as snr
import torch
from torch import Tensor
import itertools

class SignalNoiseRatio(snr):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        super().update(preds=preds, target=target)

class ScaleInvariantSignalNoiseRatio(sisnr):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        super().update(preds=preds, target=target)


class ScaleInvariantSignalNoiseRatioImprovement(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_module("s1", ScaleInvariantSignalNoiseRatio())
        self.add_module("s2", ScaleInvariantSignalNoiseRatio())

    def update(self, preds: Tensor, target: Tensor, mixture: Tensor, *args, **kwargs) -> None:
        mixture = mixture.unsqueeze(1).expand(-1, 2, -1)
        self.s1.update(target, preds)
        self.s2.update(target, mixture)

    def compute(self):
        return self.s1.compute() - self.s2.compute()
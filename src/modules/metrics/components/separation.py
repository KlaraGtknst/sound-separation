from itertools import permutations
from torchmetrics import Metric
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SISNR
from torchmetrics.audio import SignalNoiseRatio as SNR
from torchmetrics.audio import SignalDistortionRatio as SDR
from torchmetrics.audio.snr import scale_invariant_signal_noise_ratio
import torch
from torch import Tensor



class SignalNoiseRatio(SNR):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        super().update(preds=preds, target=target)


class SignalDistortionRatio(SDR):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        super().update(preds=preds, target=target)


class ScaleInvariantSignalNoiseRatio(SISNR):
    def update(self, preds: Tensor, target: Tensor, *args, **kwargs) -> None:
        super().update(preds=preds, target=target)


class ScaleInvariantSignalNoiseRatioImprovement(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("improvements", [])

    def update(self, preds: Tensor, target: Tensor, mixture: Tensor, is_supervised: Tensor = False, *args, **kwargs) -> None:
        if not is_supervised.any():
            return
        preds = preds[is_supervised]
        target = target[is_supervised]
        mixture = mixture[is_supervised]
        mixture = mixture.unsqueeze(1).expand_as(target)

        # Compute SI-SNR for each sample
        sisnr_est_values = scale_invariant_signal_noise_ratio(target, preds)  # Returns a tensor of SI-SNR values per sample
        sisnr_mix_values = scale_invariant_signal_noise_ratio(target, mixture)
        improvements = sisnr_est_values - sisnr_mix_values
        self.improvements.append(improvements)

    def compute(self):
        # Concatenate all improvements and compute the mean
        if not len(self.improvements):
            return None
        all_improvements = torch.cat(self.improvements)
        return all_improvements.mean()


class MoMi(ScaleInvariantSignalNoiseRatioImprovement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__class__.__name__ = "MoMi"

    def update(self, preds: Tensor, target: Tensor, mixture: Tensor, is_supervised: Tensor = False, *args, **kwargs) -> None:
        super().update(preds, target, mixture, torch.ones_like(is_supervised).bool())
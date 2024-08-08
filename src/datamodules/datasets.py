from typing import Any, Callable, Dict

import torch
from torch.utils.data import Dataset
import datasets
import soundfile as sf
import librosa


class BirdsetDataset(Dataset):
    def __init__(self,
                 hf_ds: datasets.Dataset,
                 sample_rate: int = 32_000,
                 transforms: Callable = None,
                 **kwargs):
        super().__init__()
        self.hf_ds = hf_ds
        self.sample_rate = sample_rate
        self.transforms = transforms

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        data = self.hf_ds[idx]
        sr = sf.info(data["filepath"]).samplerate
        wave, sr = sf.read(file=data["filepath"],
                           start=int(data["start_time"] * sr),
                           stop=int(data["end_time"] * sr))

        if wave.ndim != 1:  # ensure wave is mono
            wave = wave.swapaxes(1, 0)
            wave = librosa.to_mono(wave)

        if sr != self.sample_rate:  # ensure wave is correct sample_rate
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        data["audio"] = {"wave": wave, "sr": sr}

        if self.transforms:
            data = self.transforms(data)

        return data


class RandomDataset(Dataset):
    def __init__(
        self,
        data_path: str = None,
        transforms: Callable = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """return one training instance"""
        data = torch.rand(40)  # 40 datapoints
        label = torch.randint(low=0, high=1, size=(1,))  # for 1 class
        return {"input": data.float(), "label": label.float()}


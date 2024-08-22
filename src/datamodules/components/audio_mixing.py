import random
import glob
import os
import numpy as np
import soundfile as sf
import librosa

from src.datamodules.components.normalization import NormalizeDBFS
from src.datamodules.components.transforms import AudioTransform
from src import utils
from src.datamodules.datasets import BirdsetDataset
from copy import deepcopy
import random

log = utils.get_pylogger(__name__)

class SupervisedAudioMixing(AudioTransform):
    def __init__(self,
                 p: float,
                 audio_paths: list[str],
                 signal_ratio: float = 0.5,
                 target_dBFS: float = -20.0):

        super().__init__(p=p)

        self.audio_paths = self._find_audio_files(audio_paths)

        self.normalize = NormalizeDBFS(target_dBFS=target_dBFS)
        self.signal_ratio = signal_ratio

    def apply(self, data):
        if not data["is_supervised"]:  # only apply supervised mixtures when condition true
            return data

        audio = data["audio"]["wave"]
        sr = data["audio"]["sr"]
        background_path = random.choice(self.audio_paths)
        info = sf.info(background_path)

        background_num_samples = info.duration * info.samplerate
        random_start = 0
        if background_num_samples > len(audio):
            random_start = random.randint(0, int(background_num_samples - len(audio)))

        bg_audio, bg_sr = sf.read(background_path, start=random_start, stop=random_start + len(audio))

        if bg_audio.sum() == 0:
            print(random_start, random_start + len(audio))
            print(info.duration)

        if bg_audio.ndim != 1:  # ensure bg_audio is mono
            bg_audio = bg_audio.swapaxes(1, 0)
            bg_audio = librosa.to_mono(bg_audio)

        if sr != bg_sr:  # ensure bg_audio is correct sample_rate
            bg_audio = librosa.resample(bg_audio, orig_sr=bg_sr, target_sr=sr)

        if bg_audio.shape[0] < audio.shape[0]:
            bg_audio = np.pad(bg_audio, (0, audio.shape[0] - bg_audio.shape[0]), 'constant')
        else:
            audio = np.pad(audio, (0, bg_audio.shape[0] - audio.shape[0]), 'constant')

        bg_audio = self.normalize(bg_audio)
        mix = audio * self.signal_ratio + bg_audio * (1 - self.signal_ratio)
        data["audio"]["wave"] = np.stack([audio, bg_audio])
        data["mix"] = mix

        return data

    def set_dataset(self, dataset: BirdsetDataset):
        supervised_birds = dataset.hf_ds_supervised["filepath"]
        log.info(f"Supervised background choices: {len(self.audio_paths)}")
        log.info(f"Supervised bird choices: {len(supervised_birds)}")
        self.audio_paths += supervised_birds

    def _find_audio_files(self, audio_paths: list[str]):
        audio_files = []

        for path in audio_paths:
            if os.path.isfile(path):
                if path.endswith('.wav') or path.endswith('.ogg'):
                    audio_files.append(path)

            elif os.path.isdir(path):
                # Use glob to find all .wav and .ogg files in the specified directory and its subdirectories
                audio_files = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True) + \
                              glob.glob(os.path.join(path, '**', '*.ogg'), recursive=True)
            else:
                print(f"The provided path {path} is neither a file nor a directory.")

        return audio_files


class MixtureOfMixtures(AudioTransform):
    def __init__(self, p: float, signal_ratio: float = 0.5):
        super().__init__(p=p)
        self.dataset = None
        self.signal_ratio = signal_ratio

    def set_dataset(self, dataset: BirdsetDataset):
        self.dataset = deepcopy(dataset)
        # ensure that all augmentation after this one (including self) are not used
        l = []
        for i in self.dataset.transforms.augmentations.transforms:
            if isinstance(i, AudioTransform):
                break
            l.append(i)
        self.dataset.transforms.augmentations.transforms = l

    def apply(self, data):
        if "mix" in data:  # only apply mixing when no mix is present
            return data

        audio = data["audio"]["wave"]

        mix_data = random.choice(self.dataset)
        mix_audio = mix_data["audio"]["wave"]
        if mix_audio.shape[0] < audio.shape[0]:
            mix_audio = np.pad(mix_audio, (0, audio.shape[0] - mix_audio.shape[0]), 'constant')
        else:
            audio = np.pad(audio, (0, mix_audio.shape[0] - audio.shape[0]), 'constant')

        mix = audio * self.signal_ratio + mix_audio * (1 - self.signal_ratio)

        data["audio"]["wave"] = np.stack([audio, mix_audio])
        #data["filepath_mix"] = mix_data["filepath"]
        data["mix"] = mix

        return data

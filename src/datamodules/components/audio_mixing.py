import random
import glob
import os
import numpy as np
import soundfile as sf
import librosa

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
                 normalization: callable = None):

        super().__init__(p=p)

        self.audio_paths = self._find_audio_files(audio_paths)

        self.normalization = normalization

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

        if bg_audio.ndim != 1:  # ensure bg_audio is mono
            bg_audio = bg_audio.swapaxes(1, 0)
            bg_audio = librosa.to_mono(bg_audio)

        if sr != bg_sr:  # ensure bg_audio is correct sample_rate
            bg_audio = librosa.resample(bg_audio, orig_sr=bg_sr, target_sr=sr)

        if self.normalization:
            bg_audio = self.normalization(bg_audio)

        if bg_audio.shape[0] < audio.shape[0]:
            bg_audio = np.pad(bg_audio, (0, audio.shape[0] - bg_audio.shape[0]), 'constant')
        else:
            audio = np.pad(audio, (0, bg_audio.shape[0] - audio.shape[0]), 'constant')

        mix = audio + bg_audio
        data["audio"]["wave"] = np.stack([audio, bg_audio])
        data["mix"] = mix

        return data

    def set_dataset(self, dataset: BirdsetDataset):
        supervised_birds = dataset.hf_ds_supervised["filepath"]
        log.info(f"Supervised background choices: {len(self.audio_paths)}") # Both: 9.982
        log.info(f"Supervised bird choices: {len(supervised_birds)}") # XCM: 4.831 | XCL: 35.534
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
    def __init__(self,
                 p: float,
                 normalization: callable = None):
        super().__init__(p=p)
        self.dataset = None
        self.normalization = normalization

    def set_dataset(self, dataset: BirdsetDataset):
        self.dataset = deepcopy(dataset)
        # remove all augmentations (may include normalization)
        self.dataset.transforms.augmentations.transforms = []

    def apply(self, data):
        if "mix" in data:  # only apply mixing when no mix is present
            return data

        audio = data["audio"]["wave"]

        mix_data = random.choice(self.dataset)
        mix_audio = mix_data["audio"]["wave"]

        if self.normalization:
            mix_audio = self.normalization(mix_audio)

        if mix_audio.shape[0] < audio.shape[0]:
            mix_audio = np.pad(mix_audio, (0, audio.shape[0] - mix_audio.shape[0]), 'constant')
        else:
            audio = np.pad(audio, (0, mix_audio.shape[0] - audio.shape[0]), 'constant')

        mix = audio + mix_audio

        data["audio"]["wave"] = np.stack([audio, mix_audio])
        data["mix"] = mix

        return data

import random
import glob
import os
import numpy as np
import soundfile as sf
import librosa

from src.datamodules.components.normalization import NormalizeDBFS
from src import utils

log = utils.get_pylogger(__name__)

class AudioMixing:
    def __init__(self,
                 audio_paths: list[str],
                 signal_ratio: float = 0.5,
                 target_dBFS: float = -20.0):

        self.audio_paths = self._find_audio_files(audio_paths)

        log.info(f"Found {len(self.audio_paths)} audios")

        self.normalize = NormalizeDBFS(target_dBFS=target_dBFS)
        self.signal_ratio = signal_ratio

    def __call__(self, data):
        audio = data["audio"]["wave"]
        sr = data["audio"]["sr"]
        background_path = random.choice(self.audio_paths)
        info = sf.info(background_path)

        background_num_samples = info.duration * info.samplerate
        random_start = 0
        if background_num_samples > len(audio):
            random_start = random.randint(0, background_num_samples - len(audio))

        bg_audio, bg_sr = sf.read(background_path, start=random_start, stop=random_start + len(audio))

        if bg_audio.ndim != 1:  # ensure bg_audio is mono
            bg_audio = bg_audio.swapaxes(1, 0)
            bg_audio = librosa.to_mono(bg_audio)

        if sr != bg_sr:  # ensure bg_audio is correct sample_rate
            bg_audio = librosa.resample(bg_audio, orig_sr=bg_sr, target_sr=sr)


        if len(audio) > len(bg_audio):
            bg_audio = np.pad(bg_audio, (0, len(audio) - len(bg_audio)), 'constant')

        bg_audio = self.normalize(bg_audio)
        mix = audio * self.signal_ratio + bg_audio * (1 - self.signal_ratio)
        data["filepath_mix"] = background_path
        data["bg_audio"] = bg_audio
        data["mix"] = mix

        return data

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
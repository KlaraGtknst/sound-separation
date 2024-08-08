import numpy as np

class NormalizeRMS:
    def __call__(self, data):
        audio = data["audio"]["wave"] if type(data) is dict else data

        rms = np.sqrt(np.mean(audio**2))  # Calculate RMS
        audio /= rms

        if type(data) is dict:
            data["audio"]["wave"] = audio
            return data
        return audio

class NormalizeDBFS:
    def __init__(self, target_dBFS: float = -20.0):
        """normalizes the audios loudness to maximum dB - target_dBFS"""
        self.target_dBFS = target_dBFS

    def __call__(self, data):
        audio = data["audio"]["wave"] if type(data) is dict else data

        rms = np.sqrt(np.mean(audio**2))  # Calculate RMS
        scalar = 10 ** (self.target_dBFS / 20) / rms  # Calculate the scalar for the desired dBFS
        audio *= scalar

        if type(data) is dict:
            data["audio"]["wave"] = audio
            return data
        return audio


class InstanceZscore:
    def __init__(self, epsilon: float = 0):
        self.epsilon = epsilon

    def __call__(self, data):
        x = data["mix"]
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + self.epsilon)

        data["mix"] = x
        return data


class GlobalZscore:
    def __init__(self, dataset_mean: list, dataset_std: list, epsilon: float = 0):
        self.epsilon = epsilon
        self.dataset_mean = np.array(dataset_mean)
        self.dataset_std = np.array(dataset_std)

    def __call__(self, data):
        x = data["mix"]
        x = (x - self.dataset_mean) / (self.dataset_std + self.epsilon)

        data["mix"] = x
        return data
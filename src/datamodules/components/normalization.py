import numpy as np
import torch
from abc import ABC


class Normalization(ABC):
    def __call__(self, data):
        return data

class NormalizeRMS(Normalization):
    def __call__(self, data):
        audio = data["audio"]["wave"] if type(data) is dict else data

        rms = np.sqrt(np.mean(audio**2))  # Calculate RMS
        audio /= rms

        if type(data) is dict:
            data["audio"]["wave"] = audio
            return data
        return audio


class NormalizeDBFS(Normalization):
    def __init__(self, target_dBFS: float = -20.0, epsilon: float = 0):
        """normalizes the audios loudness to maximum dB - target_dBFS"""
        self.target_dBFS = target_dBFS
        self.epsilon = epsilon

    def __call__(self, data):
        audio = data["audio"]["wave"] if type(data) is dict else data

        rms = np.sqrt(np.mean(audio**2))  # Calculate RMS
        if rms == 0:
            return data

        scalar = 10 ** (self.target_dBFS / 20) / (rms + self.epsilon)  # Calculate the scalar for the desired dBFS
        audio *= scalar

        if type(data) is dict:
            data["audio"]["wave"] = audio
            return data
        return audio

class NormalizePeak(Normalization):
    def __init__(self, target_peak=0.2):
        self.target_peak = target_peak

    def __call__(self, data):

        input_values = data["audio"]["wave"] if type(data) is dict else data

        input_values -= np.mean(input_values, axis=-1, keepdims=True)

        # Calculate the peak normalization factor
        peak_norm = np.max(np.abs(input_values), axis=-1, keepdims=True)

        # Normalize the array to the peak value, avoiding division by zero
        input_values = np.where(
            peak_norm > 0.0,
            input_values / peak_norm,
            input_values
        )
        input_values *= self.target_peak

        if type(data) is dict:
            data["audio"]["wave"] = input_values
            return data
        return input_values

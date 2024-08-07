import numpy as np
from .transforms import AudioTransforms




class BackgroundNoise(AudioTransforms):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, inputs):
        pass

from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from abc import ABC


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig, split_name: str) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
            split_name (str): current split name (train/valid/test/predict)
        """
        order = transforms_cfg.get(split_name, None)
        augmentations = []
        if order != [] and not order:
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in order:
            augmentation = hydra.utils.instantiate(
                transforms_cfg.get(augmentation_name), _convert_="object"
            )
            augmentations.append(augmentation)
        self.augmentations = Compose(augmentations)

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        Args:
            image (Any): Input data.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """
        return self.augmentations(data=data, **kwargs)


class Compose:
    """ form torchvision"""

    def __init__(self, transforms, *args, **kwargs):
        self.transforms = transforms

    def __call__(self, data, *args, **kwargs):
        for t in self.transforms:
            data = t(data, *args, **kwargs)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class AudioTransforms(ABC):
    def __init__(self, p):
        self.p = p

    def __call__(self, inputs):
        if np.random.rand() < self.p:
            return self.apply(inputs)
        else:
            return inputs

    def apply(self, inputs):
        raise NotImplementedError()

from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        augmentations = []
        if not transforms_cfg.get("order") == [] and not transforms_cfg.get("order", None):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
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

    def __call__(self, data, **kwargs):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
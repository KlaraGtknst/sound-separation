from typing import Optional
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset
from datasets import load_dataset

from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.base_datamodule import BaseDataModule


class BirdsetDataModule(BaseDataModule):

    def __init__(self,
                 hf_dataset: DictConfig,
                 datasets: DictConfig,
                 loaders: DictConfig,
                 transforms: DictConfig,
                 **kwargs) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            hf_dataset (DictConfig): hf config used in load_dataset(**hf_dataset)
            datasets (DictConfig): Datasets config.
            loaders (DictConfig): Loaders config.
            transforms (DictConfig): Transforms config.
        """
        super().__init__(datasets=datasets, loaders=loaders, transforms=transforms)
        self.cfg_hf_dataset = hf_dataset

        mapping_cfg = hf_dataset.pop("mapping")
        filter_cfg = hf_dataset.pop("filter")

        self.hf_dataset = load_dataset(**hf_dataset)

        for i in range(len(mapping_cfg)):
            f = hydra.utils.instantiate(mapping_cfg[i])
            self.hf_dataset = self.hf_dataset.map(f, batched=True)

        for i in range(len(filter_cfg)):
            f = hydra.utils.instantiate(filter_cfg[i])
            self.hf_dataset = self.hf_dataset.filter(f)

        hf_split = self.hf_dataset.train_test_split(test_size=0.1, seed=0)
        self.hf_train_split = hf_split["train"]
        self.hf_valid_split = hf_split["test"]


    def _get_dataset_(self, split_name: str) -> Dataset:

        if self.transforms.get(split_name):
            transforms = TransformsWrapper(self.transforms.get(split_name))
        else:
            transforms = None

        cfg = self.cfg_datasets.get(split_name)
        dataset: Dataset = hydra.utils.instantiate(cfg, transforms=transforms)
        return dataset


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        if stage == "fit" and not self.train_set and not self.valid_set:
            self.train_set = self._get_dataset_("train")
            self.train_set = self._get_dataset_("valid")
        if stage == "test" and not self.test_set:
            self.train_set = self._get_dataset_("test")

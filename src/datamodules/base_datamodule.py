from collections import OrderedDict
from typing import Dict, List, Optional, Union

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.transforms import TransformsWrapper


class BaseDataModule(LightningDataModule):
    """Example of LightningDataModule for single dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return predict dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            datasets (DictConfig): Datasets config.
            loaders (DictConfig): Loaders config.
            transforms (DictConfig): Transforms config.
        """

        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = transforms
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()

    def _get_dataset_(self, split_name: str) -> Dataset:
        transforms = None
        if self.transforms.get(split_name):
            transforms = TransformsWrapper(self.transforms, split_name)

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
        if (stage == "fit" or stage == "valid") and not self.train_set and not self.valid_set:
            self.train_set = self._get_dataset_("train")
            self.valid_set = self._get_dataset_("valid")
        if stage == "test" and not self.test_set:
            self.test_set = self._get_dataset_("test")

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("valid"))

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"))

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

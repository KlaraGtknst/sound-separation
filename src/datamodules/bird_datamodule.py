from typing import Optional
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, default_collate, DataLoader
from torch.nn.functional import pad
from datasets import Audio
from datasets import load_dataset

from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.base_datamodule import BaseDataModule


class BirdsetDataModule(BaseDataModule):

    def __init__(self,
                 hf_dataset: DictConfig,
                 datasets: DictConfig,
                 loaders: DictConfig,
                 transforms: DictConfig,  # instance level
                 # augmentations: DictConfig,  # batch level
                 *args,
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

        self.mapping_cfg = hf_dataset.pop("mapping")
        self.filter_cfg = hf_dataset.pop("filter")
        self.select_columns = hf_dataset.pop("select_columns")
        self.hf_dataset = hf_dataset

        self.hf_splits = None

    def prepare_data(self) -> dict:
        hf_ds = load_dataset(**self.hf_dataset)
        hf_ds = hf_ds.select_columns(self.select_columns)
        # make sure audio decoding is off, as we load the audio later
        hf_ds = hf_ds.cast_column("audio", Audio(decode=False))

        for i in range(len(self.mapping_cfg)):
            f = hydra.utils.instantiate(self.mapping_cfg[i])
            hf_ds = hf_ds.map(f, batched=True)

        for i in range(len(self.filter_cfg)):
            f = hydra.utils.instantiate(self.filter_cfg[i])
            hf_ds = hf_ds.filter(f)

        hf_split = hf_ds.train_test_split(test_size=0.2, seed=0)

        hf_splits = {"train": hf_split["train"]}

        hf_split = hf_split["test"].train_test_split(test_size=0.5, seed=0)

        hf_splits["valid"] = hf_split["train"]
        hf_splits["test"] = hf_split["test"]

        return hf_splits


    def _get_dataset_(self, split_name: str) -> Dataset:
        transforms = None
        if self.transforms.get(split_name):
            transforms = TransformsWrapper(self.transforms, split_name)

        cfg = self.cfg_datasets.get(split_name)
        dataset: Dataset = hydra.utils.instantiate(cfg,
                                                   hf_ds=self.hf_splits[split_name],
                                                   transforms=transforms)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.hf_splits:
            self.hf_splits = self.prepare_data()
        super().setup(stage=stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"), collate_fn=_custom_collate)

    def val_dataloader(self):
        return DataLoader(self.valid_set, **self.cfg_loaders.get("valid"), collate_fn=_custom_collate)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"), collate_fn=_custom_collate)

import torch
import numpy as np

def _custom_collate(batch):
    # First, filter and convert lists and numpy arrays to tensors
    processed_batch = []
    max_length = {}
    for item in batch:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, int):
                processed_item[key] = value
            elif isinstance(value, list):
                tensor = torch.tensor(value)
                processed_item[key] = tensor
                max_length[key] = max(max_length.get(key, 0), tensor.size(0))
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                processed_item[key] = tensor
                max_length[key] = max(max_length.get(key, 0), tensor.size(0))
            elif isinstance(value, dict) and 'wave' in value and 'sr' in value:
                tensor = torch.tensor(value['wave'])
                processed_item[key] = tensor
                max_length[key] = max(max_length.get(key,0), tensor.size(0))
        processed_batch.append(processed_item)

    # Now pad arrays where necessary
    for item in processed_batch:
        for key in max_length:
            if key in item and isinstance(item[key], torch.Tensor):
                pad_size = max_length[key] - item[key].size(0)
                if pad_size > 0:
                    item[key] = pad(item[key], (0, pad_size))

    # Use the default_collate to handle the final batch conversion
    return default_collate(processed_batch)




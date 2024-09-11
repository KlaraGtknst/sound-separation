from typing import Optional
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, default_collate, DataLoader
from datasets import Audio
from datasets import load_dataset
import torch
import numpy as np

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
        self.cfg_datasets = datasets

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

        for aug in transforms.augmentations.transforms:
            if hasattr(aug, "set_dataset"):
                aug.set_dataset(dataset)

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

def _custom_collate(batch):
    # Extract the audio waves and mix arrays
    waves = [item['audio']['wave'] for item in batch]
    mixes = [item['mix'] for item in batch]

    # Get maximum length for padding
    max_wave_len = max(wave.shape[1] for wave in waves)
    max_wave_len = batch[0]["audio"]["sr"] * 10
    max_mix_len = max(len(mix) for mix in mixes)
    max_mix_len = batch[0]["audio"]["sr"] * 10

    # Pad waves
    padded_waves = []
    for wave in waves:
        # Create a new array of zeros with the maximum length
        padded_wave = np.zeros((wave.shape[0], max_wave_len))
        # Copy the original wave data into the padded array
        padded_wave[:, :wave.shape[1]] = wave
        padded_waves.append(torch.tensor(padded_wave, dtype=torch.float32))

    # Pad mixes
    padded_mixes = []
    for mix in mixes:
        # Create a new array of zeros with the maximum length
        padded_mix = np.zeros(max_mix_len)
        # Copy the original mix data into the padded array
        padded_mix[:len(mix)] = mix
        padded_mixes.append(torch.tensor(padded_mix, dtype=torch.float32))

    # Update the batch with padded data
    for i, item in enumerate(batch):
        item['audio']['wave'] = padded_waves[i]
        item['mix'] = padded_mixes[i]

    # Use default_collate to handle batching
    return default_collate(batch)




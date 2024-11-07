import h5py
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np


# Custom HDF5 Dataset class
class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        group="train",
        data_dataset="data",
        labels_dataset="labels",
        data_dtype=torch.float32,
        labels_dtype=torch.long,
    ):
        self.hdf5_file = hdf5_file
        self.group = group
        self.data_dataset = data_dataset
        self.labels_dataset = labels_dataset
        self.data_dtype = data_dtype
        self.labels_dtype = labels_dtype

        self.hdf = None  # Will be initialized in __getitem__
        self.data = None
        self.labels = None

        # Store the length without keeping the file open
        with h5py.File(self.hdf5_file, "r") as f:
            self.len = len(f[os.path.join(self.group, self.data_dataset)])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.hdf is None:
            # Open the HDF5 file when needed (in the worker process)
            self.hdf = h5py.File(self.hdf5_file, "r", swmr=True)
            self.data = self.hdf[os.path.join(self.group, self.data_dataset)]
            self.labels = (
                self.hdf[os.path.join(self.group, self.labels_dataset)]
                if self.labels_dataset is not None
                else None
            )

        data = self.data[idx]

        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(data, dtype=self.data_dtype), torch.tensor(
                label, dtype=self.labels_dtype
            )
        else:
            return torch.tensor(data, dtype=self.data_dtype)

    @staticmethod
    def get_dataloader(
        dataset_config,
        data_frac=1.0,
        batch_size=128,
        num_workers=4,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        seed=0,
        collate_fn=None,
        **kwargs,
    ):
        # Create dataset
        dataset = HDF5Dataset(**dataset_config)

        if data_frac < 1:
            dataset_len = len(dataset)
            selected_len = int(data_frac * dataset_len)
            indices = list(range(dataset_len))

            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(indices)

            selected_indices = indices[:selected_len]
            dataset = Subset(dataset, selected_indices)

        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            )
        else:
            if shuffle:
                generator = torch.Generator().manual_seed(seed)
                sampler = RandomSampler(dataset, replacement=False, generator=generator)
            else:
                sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle is handled by the sampler
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )

        return loader

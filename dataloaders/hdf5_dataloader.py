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
        label_dtype=torch.long,
        data_transform=None,
        label_transform=None,
    ):
        self.hdf5_file = hdf5_file
        self.group = group
        self.data_dataset = data_dataset
        self.labels_dataset = labels_dataset
        self.data_dtype = data_dtype
        self.label_dtype = label_dtype
        self.data_transform = data_transform
        self.label_transform = label_transform

        self.hdf = None  # Placeholder for opening the file

    def __enter__(self):
        self.hdf = h5py.File(self.hdf5_file, "r", swmr=True)
        self.data = self.hdf[os.path.join(self.group, self.data_dataset)]
        self.labels = (
            self.hdf[os.path.join(self.group, self.labels_dataset)]
            if self.labels_dataset is not None
            else None
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hdf is not None:
            self.hdf.close()

    def __len__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            return len(f[os.path.join(self.group, self.data_dataset)])

    def __getitem__(self, idx):
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf5_file, "r", swmr=True)
            self.data = self.hdf[os.path.join(self.group, self.data_dataset)]
            self.labels = (
                self.hdf[os.path.join(self.group, self.labels_dataset)]
                if self.labels_dataset is not None
                else None
            )

        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else None

        # Apply transformations if provided
        if self.data_transform:
            data = self.data_transform(torch.tensor(data, dtype=self.data_dtype))
        else:
            data = torch.tensor(data, dtype=self.data_dtype)

        if self.labels is not None:
            if self.label_transform:
                label = self.label_transform(
                    torch.tensor(label, dtype=self.label_dtype)
                )
            else:
                label = torch.tensor(label, dtype=self.label_dtype)
            return data, label
        else:
            return data

    @staticmethod  # DataLoader function for both single-GPU and DDP
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
    ):
        # Create dataset
        dataset = HDF5Dataset(
            **dataset_config,
        )

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
            # DistributedSampler for DDP
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
            )
        else:
            # Sampler for single GPU
            if shuffle:
                generator = torch.Generator().manual_seed(seed)
                sampler = RandomSampler(
                    dataset,
                    replacement=False,
                    generator=generator,
                )

            else:
                sampler = SequentialSampler(
                    dataset,
                )

        # DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle on sampler
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

        return loader

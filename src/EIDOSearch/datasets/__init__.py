#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from typing import Tuple

import numpy as np
import torch.utils.data
from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split

from EIDOSearch.datasets import transforms
from EIDOSearch.datasets.biased_mnist import BiasedMNIST
from EIDOSearch.datasets.celeba import CelebA

class MapDataset(Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    
    Args:
        dataset:
        map_fn:
    """
    
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn
    
    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    
    def __len__(self):
        return len(self.batch_sampler.sampler)
    
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def split_dataset(dataset: torch.utils.data.Dataset, percentage: float, random_seed: int = 0) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a `torch.utils.data.Dataset` instance in two non-overlapping separated `Datasets`.
    
    The split of the elements of the original `Dataset` is based on `percentage` $$\in [0, 1]$$.
    I.e. if `percentage=0.2` the first returned dataset will contain 80% of the total elements and the second 20%.

    Args:
        dataset (torch.utils.data.Dataset): `torch.utils.data.Dataset` instance to be split.
        percentage (float): percentage of elements of `dataset` contained in the second dataset.
        random_seed (int): random seed for the split generator.

    Returns:
        tuple: a tuple containing the two new datasets.

    """
    dataset_length = len(dataset)
    valid_length = int(np.floor(percentage * dataset_length))
    train_length = dataset_length - valid_length
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                generator=Generator().manual_seed(random_seed))
    
    return train_dataset, valid_dataset

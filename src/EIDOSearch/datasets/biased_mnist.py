#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
import os

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import MNIST


# ReBias Copyright (c) 2020-present NAVER Corp.
# MIT license
# Python implementation of Biased-MNIST
class BiasedMNIST(MNIST):
    """Biased-MNIST implementation in ReBias by NAVER corp with some added features.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Args:
        root (str): path to MNIST dataset.
        rho (float, optional): controls the level of the dataset bias. Defaults to 1.0. A sample is coloured with

            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`. 
        
            The number of other colours is controlled by `n_confusing_labels` (default: 9). Hence, we have:
            
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our "unbiased" setting in the test time.

            In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

        n_confusing_labels (int, optional): Defaults to 9. In the real-world cases, biases are not equally distributed, but highly unbalanced.
            We mimic the unbalanced biases by changing the number of confusing colours for each class.
            In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
            all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
            We suggest to researchers considering this benchmark for future researches.
        color_background (bool, optional): If True, inject the color bias into the image background; 
            otherwise colorize foreground. Defaults to True.

    Examples:
        ```python
        from EIDOSearch import datasets

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.BiasedMNIST(root, train=True, transform=transform,
                                       rho=0.999, color_background=True)
        dataloader = data.DataLoader(dataset=dataset, batch_size=256,
                                     shuffle=True, num_workers=4,
                                     pin_memory=True)
        ```
    """
    
    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, rho=1.0, n_confusing_labels=9,
                 color_background=True):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True
        
        self.rho = rho
        self.n_confusing_labels = n_confusing_labels
        self.color_background = color_background
        if color_background:
            self.COLOUR_MAP[-1] = [50, 255, 128]
        
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()
        
        indices = np.arange(len(self.data))
        self._shuffle(indices)
        
        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        
        self.biased_targets = self.biased_targets[indices]
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
    
    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)
    
    def _binary_to_colour(self, data, colour):
        
        # Colorize background
        if self.color_background:
            fg_data = torch.zeros_like(data)
            fg_data[data != 0] = 255
            fg_data[data == 0] = 0
            fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)
            
            bg_data = torch.zeros_like(data)
            bg_data[data == 0] = 1
            bg_data[data != 0] = 0
            bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
            bg_data = bg_data * torch.ByteTensor(colour)
            bg_data = bg_data.permute(0, 3, 1, 2)
            
            data = fg_data + bg_data
            return data.permute(0, 2, 3, 1)
        
        # Colorize digit
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 1
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=3)
        fg_data = fg_data * torch.ByteTensor(colour)
        if fg_data.sum() == 0:
            fg_data[data != 0] = 255
        fg_data = fg_data.permute(0, 3, 1, 2)
        
        return fg_data.permute(0, 2, 3, 1)
    
    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]
    
    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)
        
        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)
        
        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.rho)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))
        
        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])
        
        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)
        
        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)
        
        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])
    
    def build_biased_mnist(self):
        n_labels = self.targets.max().item() + 1
        
        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)
        
        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []
        
        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))
        
        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, int(self.biased_targets[index])

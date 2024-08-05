#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch
from torch.nn.utils import prune
from torch.nn.utils.prune import BasePruningMethod


class MagnitudePruning(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone() * torch.where(torch.abs(t) < self.threshold, torch.zeros_like(t),
                                                  torch.ones_like(t))
        return mask


class SensitivityPruning(BasePruningMethod):
    PRUNING_TYPE = 'structured'
    
    def __init__(self, module, sensitivity, threshold):
        super().__init__()
        self.module = module
        self.sensitivity = sensitivity
        self.threshold = threshold
    
    def compute_mask(self, t, default_mask):
        mask = torch.where(self.sensitivity < self.threshold, torch.zeros_like(self.sensitivity),
                           torch.ones_like(self.sensitivity))
        if isinstance(self.module, torch.nn.Linear):
            mask = mask[:, None].expand_as(default_mask)
        if isinstance(self.module, torch.nn.Conv2d):
            mask = mask[:, None, None, None].expand_as(default_mask)
        return mask * default_mask


@torch.no_grad()
def magnitude_pruning_unstructured(module, name, threshold):
    MagnitudePruning.apply(module, name, threshold)
    prune.remove(module, name)
    return module


@torch.no_grad()
def sensitivity_pruning_structured(module, name, sensitivity, threshold):
    SensitivityPruning.apply(module, name, module, sensitivity, threshold)
    prune.remove(module, name)
    return module

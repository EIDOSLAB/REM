#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from copy import deepcopy
from math import inf, isnan

import torch

from EIDOSearch.pruning.thresholding.magnitude import find_best_unstructured_magnitude_threshold
from EIDOSearch.pruning.thresholding.sensitivity import find_best_structured_sensitivity_threshold


class PlateauIdentifier:
    
    def __init__(self, model, pwe, mode='min', tolerance=0.01):
        self.model = model
        self.patience = pwe
        self.mode = mode
        self.tolerance = tolerance
        
        self.best = inf
        self.best_dict = None
        self.best_epoch = 0
        
        self.num_bad_epochs = 0
    
    @torch.no_grad()
    def __call__(self, metrics, epoch):
        current = float(metrics)
        
        if self.best_dict is None:
            self.best_dict = deepcopy(self.model.state_dict())
        
        if isnan(current):
            self.model.load_state_dict(self.best_dict)
            self._reset()
            
            return True
        
        if self._is_better(current):
            self.num_bad_epochs = 0
            if current < self.best:
                self.best = current
                self.best_dict = deepcopy(self.model.state_dict())
                self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reset()
            
            return self.best_dict, self.best_epoch
        
        return False
    
    def _is_better(self, current):
        if self.mode == "min":
            return current < (self.best + (self.best * self.tolerance))
        if self.mode == "max":
            return current > (self.best + (self.best * self.tolerance))
    
    def _reset(self):
        self.best = inf
        self.num_bad_epochs = 0
        self.best_dict = None
        self.best_epoch = 0

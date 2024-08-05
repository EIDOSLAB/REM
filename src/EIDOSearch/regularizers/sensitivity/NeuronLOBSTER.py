#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from typing import Tuple

import torch
from torch import nn, relu

from EIDOSearch.regularizers.utils import build_convbn_tuples, Hook


class NeuronLOBSTER:
    """Applies NeuronLOBSTER regularization to `model`.

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance on which to apply the regularization.
        lmbda (flaot): regularization weight.
        layers (Tuple[torch.nn.Module, ...]):  tuple of layer types on which to apply the regularization. E.g. (nn.Linear, nn.Conv2d, ...)
        momentum (float, optional): momentum factor.
        dampening (float, optional): dampening for momentum.
    """
    
    def __init__(self, model: torch.nn.Module, lmbda: float, layers: Tuple[torch.nn.Module, ...], momentum: float = 0,
                 dampening: float = 0) -> None:
        self.model = model
        self.lmbda = lmbda
        self.layers = layers
        self.momentum = momentum
        self.dampening = dampening
        
        self.eps = torch.tensor([1e-10])
        
        self.preactivations = {}
        self.momentum_buffer = {}
        self.hooks = {}
        
        self.scaler = None
        self.modules = None
        self.tuples = None
        
        self.add_hooks()
    
    def add_hooks(self):
        self.modules = list(self.model.named_modules())
        self.tuples = build_convbn_tuples(self.model)
        
        for i, (nm, m) in enumerate(self.modules):
            if isinstance(m, self.layers):
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and any(nm in i for i in self.tuples):
                    continue
                self.hooks[nm] = Hook(m, backward=True)
    
    @torch.no_grad()
    def __call__(self, rescaled=False) -> None:
        """Applies the NeuronLOBSTER regularization to `model`.

        Args:
            rescaled (bool, optional): if True the sensitivity is rescaled as `sensitivity / max(sensitivity, eps)`
            
        Returns:
            None:
        """
        for i, (nm, m) in enumerate(self.modules):
            if isinstance(m, self.layers):
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and any(nm in i for i in self.tuples):
                    continue
                
                for np, p in m.named_parameters():
                    # Weight
                    if "weight" in np:
                        preact = self.hooks[nm].output * 1. / self.scaler.get_scale() if self.scaler is not None \
                            else self.hooks[nm].output
                        
                        preact = preact.float()
                        
                        if len(preact.shape) == 4:
                            # Mean over batch-w-h
                            preact = torch.mean(preact, dim=(0, 2, 3))
                        else:
                            # Mean over batch
                            preact = torch.mean(preact, dim=0)
                        
                        # Momentum
                        if nm not in self.momentum_buffer:
                            momentum_preact = self.momentum_buffer[nm] = torch.clone(preact).detach()
                        else:
                            momentum_preact = self.momentum_buffer[nm]
                            momentum_preact.mul_(self.momentum).add_(preact, alpha=1. - self.dampening)
                        
                        sensitivity = torch.abs(momentum_preact)
                        sensitivity = sensitivity / torch.max(sensitivity, self.eps.to(sensitivity.device)) if rescaled \
                            else sensitivity
                        
                        # Insensitivity
                        insensitivity = (1. - sensitivity) if rescaled else relu(1. - sensitivity)
                        insensitivity = insensitivity.to(p.device)
                        
                        # Neuron-by-neuron (channel-by-channel) w * Ins
                        if isinstance(m, nn.Linear):
                            regu = torch.einsum('ij,i->ij', p, insensitivity)
                        elif isinstance(m, nn.Conv2d):
                            regu = torch.einsum('ijnm,i->ijnm', p, insensitivity)
                        elif isinstance(m, nn.ConvTranspose2d):
                            regu = torch.einsum('ijnm,j->ijnm', p, insensitivity)
                        elif isinstance(m, nn.BatchNorm2d):
                            regu = torch.mul(p, insensitivity)
                        else:
                            regu = torch.zeros_like(insensitivity)
                    # Bias
                    else:
                        regu = torch.mul(p, insensitivity)
                    
                    p.add_(regu, alpha=-self.lmbda)  # w - lmbd * w * Ins
    
    def set_lambda(self, lmbda: float) -> None:
        """Sets a new lambda value.

        Args:
            lmbda (flaot): lambda value.

        Returns:
            None:
        """
        self.lmbda = lmbda
    
    def set_scaler(self, scaler: torch.cuda.amp.GradScaler) -> None:
        """Sets a new `torch.cuda.amp.GradScaler`.

        Args:
            scaler (torch.cuda.amp.GradScaler): GradScaler.

        Returns:
            None:
        """
        self.scaler = scaler

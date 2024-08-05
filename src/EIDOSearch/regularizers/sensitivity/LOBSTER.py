#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
import math
from collections import defaultdict
from typing import Optional, Tuple

import torch
from torch.nn.functional import relu


class LOBSTER:
    """Applies LOBSTER regularization to `model`: https://arxiv.org/abs/2011.09905.

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance on which to apply the regularization.
        lmbda (float): regularization weight.
        layers (Tuple[torch.nn.Module, ...]): tuple of layer types on which to apply the regularization. E.g. (nn.Linear, nn.Conv2d, ...)
        
    Returns:
        None:
    """
    
    def __init__(self, model: torch.nn.Module, lmbda: float, layers: Tuple[torch.nn.Module, ...]) -> None:
        self.model = model
        self.lmbda = lmbda
        self.layers = layers
        
        self.eps = torch.tensor([1e-10])
        self.scaler = None
        
        self.adam_like = False
    
    @torch.no_grad()
    def __call__(self, rescaled: Optional[bool] = False) -> None:
        """Applies the LOBSTER regularization to `model`.

        Args:
            rescaled (bool, optional): if True the sensitivity is rescaled as `sensitivity / max(sensitivity, eps)`
            
        Returns:
            None:
        """
        i = 0
        
        for nm, m in self.model.named_modules():
            if isinstance(m, self.layers):
                for np, p in m.named_parameters():
                    if p.grad is not None:
                        grad = p.grad.detach().clone() * 1. / self.scaler.get_scale() if self.scaler is not None \
                            else p.grad.detach().clone()
                        
                        grad = grad.float()
                        
                        if self.adam_like:
                            grad = self.adam_rescale(p, grad)
                        
                        sensitivity = torch.abs(grad) / torch.max(torch.max(torch.abs(grad)),
                                                                self.eps.to(p.device)) if rescaled else torch.abs(grad)
                        
                        insensitivity = (1. - sensitivity).to(p.device) if rescaled \
                            else relu((1. - sensitivity).to(p.device))
                        
                        p.add_(p.mul(insensitivity), alpha=-self.lmbda)
                        
                        i += 1
    
    def set_lambda(self, lmbda: float) -> None:
        """Set a new lambda value.

        Args:
            lmbda (float): new lambda value.

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
    
    def set_adam_like(self, beta1, beta2, adam_eps):
        self.adam_like = True
        self.state = defaultdict(dict)
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
    
    def adam_rescale(self, p, grad):
        
        state = self.state[p]
        # Lazy state initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        exp_avgs = state['exp_avg']
        exp_avg_sqs = state['exp_avg_sq']
        
        # update the steps for each param group update
        state['step'] += 1
        # record the step after step update
        state_steps = state['step']
        
        exp_avg = exp_avgs
        exp_avg_sq = exp_avg_sqs
        step = state_steps
        
        bias_correction2 = 1 - self.beta2 ** step
        
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.adam_eps)
        
        return exp_avg / denom

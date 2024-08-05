#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from typing import Optional, Tuple

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from EIDOSearch.regularizers.sensitivity.LOBSTER import LOBSTER
from EIDOSearch.regularizers.sensitivity.NeuronLOBSTER import NeuronLOBSTER
from EIDOSearch.regularizers.utils import Hook


def accumulate_sensitivity(model, hooks, sensitivities, sensitivity_type, scaler, layers):
    for nm, m in model.named_modules():
        if isinstance(m, layers):
            if sensitivity_type == "lobster":
                for np, p in m.named_parameters():
                    grad = p.grad.detach().clone() * 1. / scaler.get_scale() if scaler is not None \
                        else p.grad.detach().clone()
                    
                    grad = torch.abs(grad.float())
                    
                    key = f"{nm}.{np}"
                    if key in sensitivities:
                        sensitivities[key] += grad
                    else:
                        sensitivities[key] = grad
            
            if sensitivity_type == "neuron-lobster":
                preact = hooks[nm].output * 1. / scaler.get_scale() if scaler is not None \
                    else hooks[nm].output
                
                preact = preact.float()
                
                if len(preact.shape) == 4:
                    # Mean over batch-w-h
                    preact = torch.mean(preact, dim=(0, 2, 3))
                else:
                    # Mean over batch
                    preact = torch.mean(preact, dim=0)
                
                sensitivity = torch.abs(preact)
                
                key = nm
                if key in sensitivities:
                    sensitivities[key] += sensitivity
                else:
                    sensitivities[key] = sensitivity
            
            if sensitivity_type == "serene-full":
                # TODO
                raise NotImplementedError
            if sensitivity_type == "serene-local":
                # TODO
                raise NotImplementedError
            if sensitivity_type == "serene-lb":
                # TODO
                raise NotImplementedError
    
    return sensitivities


def evaluate_sensitivity(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, output_index,
                         loss_function: torch.nn.Module, sensitivity_type: str, layers: Tuple[torch.nn.Module, ...],
                         device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                         rescaled: Optional[bool] = False, use_tqdm: Optional[bool] = False) -> dict:
    """Evaluates the sensitivities for a given `torch.nn.Module` instance on some `torch.utils.data.DataLoader`.

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance for which to evaluate the sensitivity.
        dataloader (torch.utils.data.DataLoader): `Dataloader` used to evaluate the sensitivity.
        loss_function (torch.nn.Module): loss function. E.g. `torch.nn.CrossEntropyLoss()`.
        sensitivity_type (str): the type of sensitivity to evaluate, can be either: lobster, neuron-lobster, serene-full, serene-local or serene-lb.
        layers (Tuple[torch.nn.Module, ...]): tuple of layer types on which to apply the regularization. E.g. (nn.Linear, nn.Conv2d, ...)
        device (torch.device): `torch.device` instance
        scaler (torch.cuda.amp.GradScaler, optional): `torch.cuda.amp.GradScaler`, required to use amp.
        rescaled (bool, optional): if True the sensitivity is calculated as max-rescaled.
        use_tqdm (bool, optional): if True shows `tqdm`'s progess bar.

    Returns:
        dict: dictionary containing, for each module, the sensitivity.

    """
    hooks = {}
    sensitivities = {}
    eps = torch.tensor([1e-10], device=next(model.parameters()).device)
    
    if sensitivity_type != "lobster":
        for nm, m in model.named_modules():
            if isinstance(m, layers):
                if sensitivity_type == "neuron-lobster":
                    hooks[nm] = Hook(m, backward=True)
                if sensitivity_type == "serene-full" or sensitivity_type == "serene-local":
                    hooks[nm] = Hook(m, backward=False)
                if sensitivity_type == "serene-lb":
                    hooks[nm] = Hook(m, backward=True)
    
    pbar = tqdm(dataloader, desc="Evaluating sensitivity", total=len(dataloader)) if use_tqdm else dataloader
    
    for data, target in pbar:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        model.zero_grad()
        
        if scaler is not None:
            with autocast():
                output = model(data)
                if output_index != -1:
                    output = output[output_index]
                loss = loss_function(output, target)
        else:
            output = model(data)
            loss = loss_function(output, target)
        
        scaler.scale(loss).backward() if scaler is not None else loss.backward()
        
        sensitivities = accumulate_sensitivity(model, hooks, sensitivities, sensitivity_type, scaler, layers)
    
    for k in sensitivities:
        sensitivities[k] /= len(dataloader)
        sensitivities[k] = sensitivities[k] / torch.max(torch.max(sensitivities[k]), eps) if rescaled \
            else sensitivities[k]
    
    return sensitivities

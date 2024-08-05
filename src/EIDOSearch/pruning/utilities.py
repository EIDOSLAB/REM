#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch
from torch import nn


@torch.no_grad()
def get_parameters_mask(model, layers):
    mask = {}
    for nm, mo in model.named_modules():
        if isinstance(mo, layers):
            for np, p in mo.named_parameters():
                name = "{}.{}".format(nm, np)
                mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))
    return mask


@torch.no_grad()
def apply_parameters_mask(model, mask):
    for nm, mo in model.named_modules():
        for np, p in mo.named_parameters():
            name = "{}.{}".format(nm, np)
            if name in mask:
                p.mul_(mask[name])


@torch.no_grad()
def apply_gradient_mask(model, mask):
    for nm, mo in model.named_modules():
        for np, p in mo.named_parameters():
            name = "{}.{}".format(nm, np)
            if name in mask:
                p.grad.data.mul_(mask[name])


@torch.no_grad()
def get_neurons_mask(model, layers):
    mask = {}
    for nm, mo in model.named_modules():
        if isinstance(mo, layers):
            for np, p in mo.named_parameters():
                name = "{}.{}".format(nm, np)
                if isinstance(mo, nn.BatchNorm2d):
                    layer_sum = p.detach().clone()
                if isinstance(mo, nn.Linear):
                    layer_sum = torch.sum(torch.abs(p), dim=1)
                if isinstance(mo, nn.modules.Conv2d):
                    layer_sum = torch.sum(torch.abs(p), dim=(1, 2, 3))
                if isinstance(mo, nn.modules.ConvTranspose2d):
                    layer_sum = torch.sum(torch.abs(p), dim=(0, 2, 3))
                
                mask[name] = torch.where(layer_sum == 0, torch.zeros_like(layer_sum), torch.ones_like(layer_sum))
    return mask


@torch.no_grad()
def apply_neurons_mask(model, mask):
    for nm, mo in model.named_modules():
        for np, p in mo.named_parameters():
            name = "{}.{}".format(nm, np)
            if name in mask:
                if isinstance(mo, nn.BatchNorm2d):
                    p.mul_(mask[name])
                if isinstance(mo, nn.Linear):
                    p.copy_(torch.einsum('ij,i->ij', p, mask[name]))
                if isinstance(mo, nn.modules.Conv2d):
                    p.copy_(torch.einsum('ijnm,i->ijnm', p, mask[name]))
                if isinstance(mo, nn.modules.ConvTranspose2d):
                    p.copy_(torch.einsum('ijnm,j->ijnm', p, mask[name]))
                    
                    
def mask_to_list(model, mask):
    mask_list = []
    
    for name, _ in model.named_parameters():
        if name in mask:
            mask_list.append(mask[name])
    return mask_list

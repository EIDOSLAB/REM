#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from typing import List

import torch.nn
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from EIDOSearch.evaluation.measures import *
from EIDOSearch.models.classification.capsule.capsule import CapsClass2d


@torch.no_grad()
def test_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, measures: list, device: torch.device,
               output_index: int,
               amp: Optional[bool] = False, use_tqdm: Optional[bool] = False) -> List[float]:
    """Evaluates the given `torch.nn.Module` instance on a given list of performance measures.
    
    A measure is a class that implements a __call__ method as:
    
    `def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> Union[int, float, list]`

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance.
        dataloader (torch.utils.data.DataLoader): `torch.utils.data.DataLoader` instance.
        measures (list): list of classes that evaluates a measure given the model's outputs and targets. See `measure.Accuracy` for reference.
        device (torch.device): `torch.device` instance.
        amp (bool, optional): if True activates `torch.cuda.amp`
        use_tqdm (bool, optional): if True shows tqdm's progress bar.

    Returns:
        List[float]: list of performance values in the order of the given `measures` list.
    """
    meters = []
    
    for i in range(len(measures)):
        meters.append(AverageMeter())
    
    if dataloader is not None:
        model.eval()
        
        if use_tqdm:
            pbar = tqdm(dataloader, desc="Testing model", total=len(dataloader))
        else:
            pbar = dataloader
        
        for batch in pbar:
            data, targets = batch[0], batch[1]
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with autocast(amp):
                outputs = model(data)
                if output_index != -1:
                    outputs = outputs[output_index]
            
            for i in range(len(measures)):
                meters[i].update(measures[i](outputs, targets), data.shape[0])
        
        model.train()
    
    return [m.avg for m in meters]


@torch.no_grad()
def architecture_stat(model: torch.nn.Module,
                      modules: Optional[Tuple[nn.Module, ...]] = (nn.Linear, nn.Conv2d)) -> dict:
    """Returns a series of architecture related statistics. E.g. number of non-zero parameters for each layer or total number of parameters for the whole model.

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance.
        modules (Tuple[nn.Module, ...], optional): tuple of `torch.nn.Module` that the method should consider. E.g. `nn.Linear` or `nn.Conv2d`.

    Returns:
        dict: - `layer_connection_max`:         Maximum number of connections for each layer
              - `layer_connection_mean`:        Mean number of connections for each layer
              - `layer_connection_min`:         Minimum number of connections for each layer
              - `layer_connection_std`:         STD of the connection for each layer
              - `layer_neuron_non_zero`:        Number of non-zero neurons for each layer
              - `layer_neuron_non_zero_perc`:   Percentage of non-zero neurons for each layer
              - `layer_neuron_ratio`:           Compression ratio of the neurons for each layer
              - `layer_param_non_zero`:         Number of non-zero parameters for each layer
              - `layer_param_non_zero_perc`:    Percentage of non-zero parameters for each layer
              - `layer_param_ratio`:            Compression ratio of the parameters for each layer
              - `network_neuron_non_zero`:      Number of non-zero neurons for the whole model
              - `network_neuron_non_zero_perc`: Percentage of non-zero neurons for the whole model
              - `network_neuron_ratio`:         Compression ration of the neurons for the whole model
              - `network_neuron_total`:         Total number of neurons for the whole model
              - `network_param_non_zero`:       Number of non-zero parameters for the whole model
              - `network_param_non_zero_perc`:  Percentage of non-zero parameters for the whole model
              - `network_param_ratio`:          Compression ratio of the parameters for the whole model
              - `network_param_total`:          Total number of parameters for the whole model
    """
    layer_param_total = {}
    layer_param_non_zero = {}
    layer_param_non_zero_perc = {}
    layer_param_ratio = {}
    
    layer_neuron_total = {}
    layer_neuron_non_zero = {}
    layer_neuron_non_zero_perc = {}
    layer_neuron_ratio = {}
    
    layer_connection_max = {}
    layer_connection_min = {}
    layer_connection_mean = {}
    layer_connection_std = {}
    
    for nm, mo in model.named_modules():
        if isinstance(mo, modules):
            for np, p in mo.named_parameters():
                name = "{}.{}".format(nm, np)
                layer_param_total[name] = p.numel()
                layer_param_non_zero[name] = torch.nonzero(p, as_tuple=False).shape[0]
                layer_param_non_zero_perc[name] = layer_param_non_zero[name] / layer_param_total[name] * 100
                layer_param_ratio[name] = (layer_param_total[name] / layer_param_non_zero[name]) \
                    if (layer_param_non_zero[name] != 0) else -1
                
                if 'weight' in name:
                    if isinstance(mo, nn.Linear):
                        params_sum = torch.abs(p).sum(dim=1)
                    elif isinstance(mo, nn.Conv2d):
                        params_sum = torch.abs(p).sum(dim=(1, 2, 3))
                    elif isinstance(mo, nn.ConvTranspose2d):
                        params_sum = torch.abs(p).sum(dim=(0, 2, 3))
                    elif isinstance(mo, nn.BatchNorm2d):
                        params_sum = torch.abs(p)
                    
                    if not isinstance(mo, CapsClass2d):
                        
                        layer_neuron_total[name] = params_sum.shape[0]
                        layer_neuron_non_zero[name] = torch.nonzero(params_sum, as_tuple=False).shape[0]
                        layer_neuron_non_zero_perc[name] = layer_neuron_non_zero[name] / layer_neuron_total[name] * 100
                        layer_neuron_ratio[name] = (layer_neuron_total[name] / layer_neuron_non_zero[name]) \
                            if (layer_neuron_non_zero[name] != 0) else -1
                        
                        connections_count = torch.where(p != 0, torch.ones_like(p), torch.zeros_like(p))
                        
                        if isinstance(mo, nn.modules.Linear):
                            connections_count = connections_count.sum(dim=1)
                        elif isinstance(mo, nn.modules.Conv2d):
                            connections_count = connections_count.sum(dim=(1, 2, 3))
                        elif isinstance(mo, nn.modules.ConvTranspose2d):
                            connections_count = connections_count.sum(dim=(0, 2, 3))
                        else:
                            connections_count = connections_count
                        
                        connections_count = connections_count[connections_count != 0]
                        
                        layer_connection_max[name] = connections_count.max() if connections_count.numel() > 0 else 0
                        layer_connection_min[name] = connections_count.min() if connections_count.numel() > 0 else 0
                        layer_connection_mean[name] = connections_count.mean() if connections_count.numel() > 0 else 0
                        layer_connection_std[name] = connections_count.std() if connections_count.numel() > 0 else 0
    
    network_param_total = sum(layer_param_total.values())
    network_param_non_zero = sum(layer_param_non_zero.values())
    network_param_non_zero_perc = network_param_non_zero / network_param_total * 100 if (network_param_total != 0) \
        else -1
    network_param_ratio = network_param_total / network_param_non_zero if (network_param_non_zero != 0) else -1
    
    network_neuron_total = sum(layer_neuron_total.values())
    network_neuron_non_zero = sum(layer_neuron_non_zero.values())
    
    network_neuron_non_zero_perc = network_neuron_non_zero / network_neuron_total * 100 if (network_neuron_total != 0) \
        else -1
    network_neuron_ratio = network_neuron_total / network_neuron_non_zero if (network_neuron_non_zero != 0) else -1
    
    return {"layer_connection_max":         layer_connection_max,
            "layer_connection_mean":        layer_connection_mean,
            "layer_connection_min":         layer_connection_min,
            "layer_connection_std":         layer_connection_std,
            "layer_neuron_non_zero":        layer_neuron_non_zero,
            "layer_neuron_non_zero_perc":   layer_neuron_non_zero_perc,
            "layer_neuron_ratio":           layer_neuron_ratio,
            "layer_neuron_total":           layer_neuron_total,
            "layer_param_non_zero":         layer_param_non_zero,
            "layer_param_non_zero_perc":    layer_param_non_zero_perc,
            "layer_param_ratio":            layer_param_ratio,
            "layer_param_total":            layer_param_total,
            "network_neuron_non_zero":      network_neuron_non_zero,
            "network_neuron_non_zero_perc": network_neuron_non_zero_perc,
            "network_neuron_ratio":         network_neuron_ratio,
            "network_neuron_total":         network_neuron_total,
            "network_param_non_zero":       network_param_non_zero,
            "network_param_non_zero_perc":  network_param_non_zero_perc,
            "network_param_ratio":          network_param_ratio,
            "network_param_total":          network_param_total}

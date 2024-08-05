#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from copy import deepcopy

import numpy
import torch
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def get_neurons_clusters(tensor, twt, sensitivity, s):
    t = deepcopy(tensor).detach()
    t = t.reshape(t.shape[0], -1)
    
    lut = {}  # look-up-table new:old
    reverse_lut = {}
    sub = 0
    
    for i in range(t.shape[0]):
        if torch.sum(torch.abs(t[i])) == 0:
            sub += 1
            lut[i] = -1
            reverse_lut[i] = -1
        else:
            lut[i - sub] = i
            reverse_lut[i] = i - sub
    
    weights = sensitivity[torch.sum(torch.abs(t), dim=1) != 0]
    t = t[torch.sum(torch.abs(t), dim=1) != 0]
    t_cpu = t.cpu()
    weights_cpu = weights.cpu()
    
    bound = 0
    num = 0
    scaled_t = t * weights.to(t).view(weights.shape[0], -1)
    for i in range(scaled_t.shape[0]):
        for j in range(i + 1, scaled_t.shape[0]):
            bound += torch.mean(torch.pow(scaled_t[i] - scaled_t[j], 2))
            num += 1
    bound /= num
    
    bound *= twt
    
    if s is not None:
        s += ("K range {} - {}\n".format(1, scaled_t.shape[0]))
    
    best_K = scaled_t.shape[0]
    
    for k in reversed(range(1, scaled_t.shape[0])):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(t_cpu, sample_weight=weights_cpu)
        
        sse = 0
        for c in range(k):
            cluster_elements = numpy.array([n for n in numpy.where(kmeans.labels_ == c)[0]])
            
            if len(cluster_elements):
                highest_sens = (weights == torch.max(weights[cluster_elements])).nonzero(as_tuple=False)[0].item()
                for n in cluster_elements:
                    sse += torch.sum(torch.pow(scaled_t[highest_sens] - scaled_t[n], 2))
        
        mse = sse / t.shape[0]
        
        if s is not None:
            s += ("K {} - mse {} - bound {}\n".format(k, mse, bound))
        
        if mse > bound:
            break
        else:
            best_K = k
    
    kmeans = KMeans(n_clusters=best_K, random_state=0).fit(t_cpu, sample_weight=weights_cpu)
    
    return kmeans, lut, len(numpy.unique(kmeans.labels_)), s


@torch.no_grad()
def prune_clusters(tensor, kmeans, n_clusters, lut, sensitivity, s=None):
    prune_mask = torch.ones(tensor.shape[0]).to(tensor)
    pruned_neurons = 0
    
    if n_clusters != tensor.shape[0]:
        for c in range(n_clusters):
            neurons_in_cluster = numpy.array([lut[n] for n in numpy.where(kmeans.labels_ == c)[0]])
            
            if len(neurons_in_cluster):
                highest_sens = torch.argmax(sensitivity[neurons_in_cluster])
                neurons_to_remove = neurons_in_cluster[neurons_in_cluster != neurons_in_cluster[highest_sens]]
                
                tensor[neurons_to_remove] = tensor[neurons_to_remove].mul(0.)
                prune_mask[neurons_to_remove] = prune_mask[neurons_to_remove].mul(0.)
                pruned_neurons += len(neurons_to_remove)
    
    if s is not None:
        s += ("Pruning {} neurons\n".format(pruned_neurons))
    
    return prune_mask, s


@torch.no_grad()
def cluster_and_prune(tensor, sensitivity, twt, s=None):
    kmeans, lut, best_K, s = get_neurons_clusters(tensor, twt, sensitivity, s)
    prune_mask, s = prune_clusters(tensor, kmeans, best_K, lut, sensitivity, s)
    return prune_mask, best_K, s


class Scheduler(object):
    
    def __init__(self, model, sensitivity_optimizer, layers, valid_loader, loss_function, twt, device, amp,
                 bn_prune=False, out_file_path=None, task="classification"):
        self.model = model
        self.sensitivity_optimizer = sensitivity_optimizer
        self.layers = layers
        
        self.valid_loader = valid_loader
        self.loss_function = loss_function
        
        self.twt = twt
        
        self.device = device
        self.amp = amp
        
        self.bn_prune = bn_prune
        
        self.bound = 0
        
        self.task = task
        
        self.out_file_path = out_file_path
        
        self.tag = "-- Clustering Scheduler --"
    
    @torch.no_grad()
    def step(self):
        
        s = None
        
        layers = list(reversed(list(self.model.named_modules())[:-1]))
        pbar = tqdm(layers, total=len(layers))
        
        sensitivity, _ = self.sensitivity_optimizer.evaluate_sensitivity(self.valid_loader, self.loss_function,
                                                                         self.device)
        
        for i, (layer_name, layer) in enumerate(pbar):
            pbar.set_description("Pruning {}".format(layer_name))
            
            if isinstance(layer, self.layers):
                
                if self.out_file_path is not None:
                    s = ""
                
                if s is not None:
                    title = "{} K search for {}\n".format(self.tag, layer_name)
                    s += ("#" * len(title) + "\n")
                    s += title
                    s += ("#" * len(title) + "\n")
                
                if self.bn_prune and isinstance(layer, (nn.modules.Conv2d, nn.modules.ConvTranspose2d)):
                    if isinstance(layers[i - 1][1], nn.modules.BatchNorm2d):
                        if prune_mask is not None:
                            for n_p, p in layer.named_parameters():
                                if "weight" in n_p:
                                    if prune_mask.shape[0] != 1:
                                        if isinstance(layer, nn.modules.Conv2d):
                                            p.copy_(torch.einsum(
                                                'ijnm,i->ijnm',
                                                p,
                                                prune_mask))
                                        if isinstance(layer, nn.modules.ConvTranspose2d):
                                            p.copy_(torch.einsum(
                                                'ijnm,j->ijnm',
                                                p,
                                                prune_mask))
                                    else:
                                        p.copy_(torch.mul(p, prune_mask))
                                else:
                                    p.copy_(torch.mul(p, prune_mask))
                            
                            continue
                
                layer_sensitivity = sensitivity[layer_name].detach().clone()
                
                prune_mask, best_K, s = cluster_and_prune(layer.weight, layer_sensitivity, self.twt, s)
                
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.copy_(torch.mul(layer.bias, prune_mask))
                
                if s is not None:
                    s += ("{} chosen K: {}\n".format(self.tag, best_K))
                
                if s is not None:
                    s += "=" * 1000 + "\n\n"
                    with open(self.out_file_path, "a") as file:
                        file.write(s)
    
    def set_twt(self, twt):
        self.twt = twt

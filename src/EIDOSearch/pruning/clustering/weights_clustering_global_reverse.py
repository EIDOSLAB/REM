#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from copy import deepcopy

import numpy
import torch
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm

from EIDOSearch.evaluation import test_model


@torch.no_grad()
def get_neurons_clusters(tensor, n_clusters, sensitivity):
    t = deepcopy(tensor).detach().cpu()
    t = t.reshape(t.shape[0], -1)
    
    lut = {}
    sub = 0
    
    for i in range(t.shape[0]):
        if torch.sum(torch.abs(t[i])) == 0:
            sub += 1
            lut[i] = -1
        else:
            lut[i - sub] = i
    
    weights = sensitivity[(t != 0).view(-1)]
    t = t[torch.sum(torch.abs(t), dim=1) != 0]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(t, sample_weight=weights)
    
    return kmeans, lut


@torch.no_grad()
def prune_clusters(tensor, kmeans, n_clusters, lut, sensitivity, s=None):
    prune_mask = torch.ones(tensor.shape[0]).to(tensor)
    pruned_neurons = 0
    
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
def cluster_and_prune(tensor, n_clusters, sensitivity, s=None):
    kmeans, lut = get_neurons_clusters(tensor, n_clusters, sensitivity)
    prune_mask, s = prune_clusters(tensor, kmeans, n_clusters, lut, sensitivity, s)
    return prune_mask, s


class Scheduler(object):
    
    def __init__(self, model, sensitivity_optimizer, layers, valid_loader, loss_function, twt, device, amp,
                 bn_prune=False, out_file_path=None, task="classification"):
        self.model = model
        self.sensitivity_optimizer = sensitivity_optimizer
        self.layers = layers
        self.starting_state = None
        
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
    def step(self, initial_performance):
        
        s = None
        prune_mask = None
        
        layers = list(reversed(list(self.model.named_modules())[:-1]))
        pbar = tqdm(layers, total=len(layers))
        
        self.bound = (100 - initial_performance[0]) + ((100 - initial_performance[0]) * self.twt)
        
        sensitivity = self.sensitivity_optimizer.evaluate_sensitivity(None, self.valid_loader,
                                                                      self.loss_function, self.device, self.amp)
        
        for i, (layer_name, layer) in enumerate(pbar):
            if self.out_file_path is not None:
                s = ""
                s += ("{} Bound {}\n".format(self.tag, self.bound))
            pbar.set_description("Pruning {}".format(layer_name))
            
            if isinstance(layer, self.layers):
                
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
                            
                            # pruned_performance = test_model(self.model, self.loss_function, self.valid_loader,
                            #                                 self.device, self.task, self.amp)
                            # if s is not None:
                            #     s += ("{} error: {}\n".format(self.tag, 100 - pruned_performance[0]))
                            #     s += "=" * 1000 + "\n\n"
                            #     with open(self.out_file_path, "a") as file:
                            #         file.write(s)
                            
                            continue
                
                self.starting_state = deepcopy(self.model.state_dict())
                layer_sensitivity = sensitivity[layer_name].detach().clone()
                
                t = deepcopy(layer.weight).detach().cpu()
                t = t.reshape(t.shape[0], -1)
                max_k = t[torch.sum(torch.abs(t), dim=1) != 0].shape[0]
                
                best_K = max_k
                
                for n_clusters in reversed(range(2, max_k)):  # from max-1 to 2
                    
                    s += ("{} trying K {} -> ".format(self.tag, n_clusters))
                    
                    prune_mask, s = cluster_and_prune(layer.weight, n_clusters, layer_sensitivity, s)
                    
                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.copy_(torch.mul(layer.bias, prune_mask))
                    
                    pruned_performance = test_model(self.model, self.loss_function, self.valid_loader, self.device,
                                                    self.task, self.amp)
                    if s is not None:
                        s += ("{} error: {}\n".format(self.tag, 100 - pruned_performance[0]))
                    
                    if 100 - pruned_performance[0] <= self.bound:
                        best_K = n_clusters
                        best_perf = pruned_performance
                        self.model.load_state_dict(self.starting_state)
                    else:
                        self.model.load_state_dict(self.starting_state)
                        break
                
                if best_K != t.shape[0]:
                    prune_mask, s = cluster_and_prune(layer.weight, best_K, layer_sensitivity, s)
                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.copy_(torch.mul(layer.bias, prune_mask))
                else:
                    # best_perf = test_model(self.model, self.loss_function, self.valid_loader, self.device, self.task, self.amp)
                    prune_mask = torch.ones(layer.weight.shape[0]).to(layer.weight)
                
                if s is not None:
                    # s += ("{} chosen K {} with error {}\n".format(self.tag, best_K,
                    #                                               100 - best_perf[0] if best_K != max_k else "didn't prune"))
                    s += ("{} chosen K {}\n".format(self.tag, best_K))
                    
                    s += "=" * 1000 + "\n\n"
                    with open(self.out_file_path, "a") as file:
                        file.write(s)
    
    def set_twt(self, twt):
        self.twt = twt

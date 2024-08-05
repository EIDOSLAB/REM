#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from copy import deepcopy
from math import inf
from typing import Optional, Tuple

import torch

from EIDOSearch.evaluation import test_model
from EIDOSearch.pruning.thresholding.methods import magnitude_pruning_unstructured


def _update_magnitude_search_range(mode, thresholded_model_measure, measure_boundary,
                                   max_value, mean_value, min_value):
    # I want the biggest T that MAXIMIZE the measure
    if mode == "max":
        if measure_boundary > thresholded_model_measure:
            return mean_value, (mean_value + min_value) / 2, min_value
        else:
            return max_value, (max_value + mean_value) / 2, mean_value
    
    # I want the biggest T that MINIMIZE the measure
    else:
        if measure_boundary < thresholded_model_measure:
            return mean_value, (mean_value + min_value) / 2, min_value
        else:
            return max_value, (max_value + mean_value) / 2, mean_value


def _find_max_mean_min_magnitude(model, params):
    max_magnitude = 0
    mean_magnitude = 0
    min_magnitude = float(inf)
    parameters_counter = 0

    for np, p in model.named_parameters():
        for param in params:
            if param in np:
                magnitude = torch.abs(p)
                local_max_magnitude = torch.max(magnitude)
                local_min_magnitude = torch.min(magnitude)
                
                if max_magnitude < local_max_magnitude:
                    max_magnitude = local_max_magnitude
                
                if min_magnitude > local_min_magnitude:
                    min_magnitude = local_min_magnitude
                
                mean_magnitude += torch.sum(magnitude)
                parameters_counter += torch.numel(magnitude)
    
    return max_magnitude.item(), (mean_magnitude / parameters_counter).item(), min_magnitude.item()


def find_best_unstructured_magnitude_threshold(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, measure,
                                               twt: float, mode: str, scope: str, layers: Tuple[torch.nn.Module, ...],
                                               device: torch.device, output_index, amp: Optional[bool] = False,
                                               tqdm: Optional[bool] = False, params: Tuple[str, ...] = ["weight"]):
    # Define the measure boundary
    base_measure = test_model(model, dataloader, [measure], device, output_index, amp, tqdm)
    measure_boundary = base_measure[0] * (1 + twt)
    
    eps = 1e-10
    max_same_iterations = 10
    
    if scope == "global":
        # Save the model starting state
        starting_state = deepcopy(model.state_dict())
        
        # Find the limits to the magnitude range and the average value
        max_magnitude, mean_magnitude, min_magnitude = _find_max_mean_min_magnitude(model, params)
        previous_mean_magnitude = float(inf)
        previous_thresholded_model_measure = float(inf)
        last_ok_threshold = float(0)
        same_measure_count = 0
        
        while True:
            print(f"Range {max_magnitude:.4f} "
                  f"{mean_magnitude:.4f} "
                  f"{min_magnitude:.4f}")
            
            # Prune model
            for m in model.modules():
                if isinstance(m, layers):
                    params_names = []
                    for name, param in m.named_parameters():
                        if param.grad is not None:
                            params_names.append(name)
                    for p in params:
                        if p in params_names:
                            magnitude_pruning_unstructured(m, p, mean_magnitude)
            
            # Test model and reload state
            thresholded_model_measure = test_model(model, dataloader, [measure], device, output_index, amp, tqdm)
            thresholded_model_measure = thresholded_model_measure[0]
            
            if (mode == "max" and thresholded_model_measure >= measure_boundary) \
                    or (mode == "min" and thresholded_model_measure <= measure_boundary):
                last_ok_threshold = mean_magnitude
            
            print(f"Threshold {mean_magnitude:.4f} -> "
                  f"{thresholded_model_measure} vs "
                  f"{measure_boundary}")
            
            model.load_state_dict(starting_state)
            
            # Update search boundaries
            max_magnitude, mean_magnitude, min_magnitude = _update_magnitude_search_range(mode,
                                                                                          thresholded_model_measure,
                                                                                          measure_boundary,
                                                                                          max_magnitude,
                                                                                          mean_magnitude, min_magnitude)
            
            if thresholded_model_measure == previous_thresholded_model_measure:
                same_measure_count += 1
            else:
                same_measure_count = 0
            
            if abs(previous_mean_magnitude - mean_magnitude) < eps or same_measure_count == max_same_iterations:
                
                if mean_magnitude != last_ok_threshold:
                    mean_magnitude = last_ok_threshold
                
                for m in model.modules():
                    if isinstance(m, layers):
                        params_names = []
                        for name, param in m.named_parameters():
                            if param.grad is not None:
                                params_names.append(name)
                        for p in params:
                            if p in params_names:
                                magnitude_pruning_unstructured(m, p, mean_magnitude)
                
                print(last_ok_threshold)
                
                return mean_magnitude
            
            previous_mean_magnitude = mean_magnitude
            previous_thresholded_model_measure = thresholded_model_measure
    
    if scope == "local":
        Ts = []
        
        for n, m in model.named_modules():
            if isinstance(m, layers):
                
                print(n)
                
                # Save the model starting state
                starting_state = deepcopy(model.state_dict())
                
                # Find the limits to the magnitude range and the average value
                max_magnitude, mean_magnitude, min_magnitude = _find_max_mean_min_magnitude(m)
                previous_mean_magnitude = float(inf)
                previous_thresholded_model_measure = float(inf)
                last_ok_threshold = float(0)
                same_measure_count = 0
                
                while True:
                    print(f"Range {max_magnitude:.4f} "
                          f"{mean_magnitude:.4f} "
                          f"{min_magnitude:.4f}")
                    
                    # Prune module
                    magnitude_pruning_unstructured(m, "weight", mean_magnitude)
                    
                    # Test model and reload state
                    thresholded_model_measure = test_model(model, dataloader, [measure], device, output_index, amp,
                                                           tqdm)
                    thresholded_model_measure = thresholded_model_measure[0]
                    
                    if (mode == "max" and thresholded_model_measure >= measure_boundary) \
                            or (mode == "min" and thresholded_model_measure <= measure_boundary):
                        last_ok_threshold = mean_magnitude
                    
                    print(f"Threshold {mean_magnitude:.4f} -> "
                          f"{thresholded_model_measure} vs "
                          f"{measure_boundary}")
                    
                    model.load_state_dict(starting_state)
                    
                    # Update search boundaries
                    max_magnitude, mean_magnitude, min_magnitude = _update_magnitude_search_range(mode,
                                                                                                  thresholded_model_measure,
                                                                                                  measure_boundary,
                                                                                                  max_magnitude,
                                                                                                  mean_magnitude,
                                                                                                  min_magnitude)
                    
                    if thresholded_model_measure == previous_thresholded_model_measure:
                        same_measure_count += 1
                    else:
                        same_measure_count = 0
                    
                    if abs(previous_mean_magnitude - mean_magnitude) < eps or same_measure_count == max_same_iterations:
                        
                        if mean_magnitude != last_ok_threshold:
                            mean_magnitude = last_ok_threshold
                        
                        magnitude_pruning_unstructured(m, "weight", mean_magnitude)
                        Ts.append(mean_magnitude)
                        
                        print(last_ok_threshold)
                        
                        break
                    
                    previous_mean_magnitude = mean_magnitude
                    previous_thresholded_model_measure = thresholded_model_measure
        
        return Ts

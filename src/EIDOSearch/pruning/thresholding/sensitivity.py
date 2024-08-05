#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from copy import deepcopy
from math import inf
from typing import Optional, Tuple

import torch

from EIDOSearch.evaluation import test_model
from EIDOSearch.pruning.thresholding.methods import sensitivity_pruning_structured
from EIDOSearch.regularizers import evaluate_sensitivity


def _find_max_mean_min_sensitivity(sensitivity):
    mean_idx = ((len(sensitivity) - 1) + 0) // 2
    return [sensitivity[0], 0], [sensitivity[mean_idx], mean_idx], [sensitivity[-1], len(sensitivity) - 1]


def _update_sensitivity_search_range(mode, thresholded_model_measure, measure_boundary, sensitivity,
                                     max_value, mean_value, min_value):
    # I want the biggest T that MAXIMIZE the measure
    if mode == "max":
        if measure_boundary > thresholded_model_measure:
            median_idx = (mean_value[1] + min_value[1]) // 2
            return mean_value, [sensitivity[median_idx], median_idx], min_value
        else:
            median_idx = (max_value[1] + mean_value[1]) // 2
            return max_value, [sensitivity[median_idx], median_idx], mean_value
    
    # I want the biggest T that MINIMIZE the measure
    else:
        if measure_boundary < thresholded_model_measure:
            median_idx = (mean_value[1] + min_value[1]) // 2
            return mean_value, [sensitivity[median_idx], median_idx], min_value
        else:
            median_idx = (max_value[1] + mean_value[1]) // 2
            return max_value, [sensitivity[median_idx], median_idx], mean_value


def find_best_structured_sensitivity_threshold(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                               sensitivity, measure, loss_function, twt: float, mode: str, scope: str,
                                               layers: Tuple[torch.nn.Module, ...], device: torch.device, scaler,
                                               rescaled, output_index, amp: Optional[bool] = False,
                                               tqdm: Optional[bool] = False):
    # Define the measure boundary
    base_measure = test_model(model, dataloader, [measure], device, output_index, amp, tqdm)
    measure_boundary = base_measure[0] * (1 + twt)
    
    eps = 1e-10
    max_same_iterations = 10
    
    if scope == "global":
        pass
    
    if scope == "local":
        Ts = []
        
        for n, m in reversed(list(model.named_modules())[:-1]):
            if isinstance(m, layers):
                
                print(n)
                
                # TODO potentially we could evaluate S only for the current param
                sensitivities = evaluate_sensitivity(model, dataloader, loss_function, sensitivity, layers, device,
                                                     scaler, rescaled, tqdm)
                layer_sensitivity = torch.sort(torch.unique(sensitivities[n]), descending=True)[0]
                
                # Save the model starting state
                starting_state = deepcopy(model.state_dict())
                
                # Find the limits to the magnitude range and the average value
                max_sensitivity, mean_sensitivity, min_sensitivity = _find_max_mean_min_sensitivity(layer_sensitivity)
                previous_mean_sensitivity = (float(inf), -1)
                previous_thresholded_model_measure = float(inf)
                last_ok_threshold = (0, -1)
                same_measure_count = 0
                
                while True:
                    print(f"Range {max_sensitivity[0]:.4f}({max_sensitivity[1]}) "
                          f"{mean_sensitivity[0]:.4f}({mean_sensitivity[1]}) "
                          f"{min_sensitivity[0]:.4f}({min_sensitivity[1]})")
                    
                    # Prune module
                    sensitivity_pruning_structured(m, "weight", sensitivities[n], mean_sensitivity[0])
                    
                    # Test model and reload state
                    thresholded_model_measure = test_model(model, dataloader, [measure], device, output_index, amp,
                                                           tqdm)
                    thresholded_model_measure = thresholded_model_measure[0]
                    
                    if (mode == "max" and thresholded_model_measure >= measure_boundary) \
                            or (mode == "min" and thresholded_model_measure <= measure_boundary):
                        last_ok_threshold = mean_sensitivity
                    
                    print(f"Threshold {mean_sensitivity[0]:.4f}({mean_sensitivity[1]}) -> "
                          f"{thresholded_model_measure} vs "
                          f"{measure_boundary}")
                    
                    model.load_state_dict(starting_state)
                    
                    # Update search boundaries
                    max_sensitivity, \
                    mean_sensitivity, \
                    min_sensitivity = _update_sensitivity_search_range(mode, thresholded_model_measure,
                                                                       measure_boundary, layer_sensitivity,
                                                                       max_sensitivity, mean_sensitivity,
                                                                       min_sensitivity)
                    
                    if thresholded_model_measure == previous_thresholded_model_measure:
                        same_measure_count += 1
                    else:
                        same_measure_count = 0
                    
                    if abs(previous_mean_sensitivity[0] - mean_sensitivity[0]) < eps \
                            or same_measure_count == max_same_iterations:
                        
                        if mean_sensitivity[1] != last_ok_threshold[1]:
                            mean_sensitivity = last_ok_threshold
                        
                        Ts.append(mean_sensitivity[0])
                        sensitivity_pruning_structured(m, "weight", sensitivities[n], mean_sensitivity[0])
                        
                        print(last_ok_threshold)
                        
                        break
                    
                    previous_mean_sensitivity = mean_sensitivity
                    previous_thresholded_model_measure = thresholded_model_measure
        
        return Ts

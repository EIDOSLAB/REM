#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from typing import List, Optional

import torch
from torch import Tensor


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        masks,
        dampening: float,
        nesterov: bool):
    
    for i, param in enumerate(params):
        
        d_p = d_p_list[i]
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        
        if momentum != 0:
            buf = momentum_buffer_list[i]
            
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
                
        if masks is not None:
            d_p.mul_(masks[i])
        
        param.add_(d_p, alpha=-lr)

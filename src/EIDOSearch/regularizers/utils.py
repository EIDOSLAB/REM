#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from typing import List, Tuple

import torch.nn
from torch import nn, fx
from torch.fx.experimental.optimization import matches_module_pattern
from torchvision.models.densenet import _DenseLayer


class Hook:
    """Registers an hook to the specified `torch.nn.Module`.

    Args:
        module (torch.nn.Module): the module to which the hook is attached.
        backward (bool): if True the hook will be a `forward_hook` and gather the `input` and `output` of the module.
        If False the hook will be a `backward_hook` and gather the `grad_input` and `grad_output` of the module.
    """
    
    def __init__(self, module, backward=False) -> None:
        self.module = module
        self.input = None
        self.output = None
        self.backward = backward
        
        if not self.backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.input = input
        self.output = output if not self.backward else output[0]
    
    def close(self) -> None:
        """Remove the hook from `module`.

        Returns:
            None:

        """
        self.hook.remove()


def build_convbn_tuples(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """

    Args:
        model:

    Returns:
        List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """
    convbn_tuples = []
    
    try:
        patterns = [(nn.Linear, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.ConvTranspose2d, nn.BatchNorm2d)]
        fx_model = fx.symbolic_trace(model)
        modules = dict(fx_model.named_modules())
        
        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:
                        continue
                    convbn_tuples.append([node.args[0].target, node.target])
    
    except Exception:
        last_module = None
        
        for nm, m in model.named_modules():
            if isinstance(m, _DenseLayer):
                last_module = None
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                last_module = (nm, m)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if last_module is not None and last_module[1].weight.shape[0] == m.weight.shape[0]:
                    convbn_tuples.append([last_module[0], nm])
    
    return convbn_tuples

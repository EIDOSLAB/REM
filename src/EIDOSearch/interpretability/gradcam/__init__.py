#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).
from typing import Union, Tuple

import torch
import torch.nn.functional as F


class Gradcam:
    """ Class for generating Grad-CAM activation maps, reference https://arxiv.org/pdf/1610.02391.pdf.

    Args:
        model (torch.nn.module): `torch.nn.module` instance.
        y_name (str): output layer name.
        A_name (str): target layer name.
    """
    
    def __init__(self, model: torch.nn.Module, y_name: str, A_name: str) -> None:
        self.activations = {}
        self.hooks = []
        self.model = model
        self.y_name = y_name
        self.A_name = A_name
    
    def register_hooks(self) -> None:
        """Register the `Gradcam` hooks to `model`.
        
        Returns:
            None:
        """
        
        def record_activations(name, output):
            self.activations[name] = output
        
        self.hooks = []
        for n, m in self.model.named_modules():
            if n == self.y_name or n == self.A_name:
                h = m.register_forward_hook(lambda m, i, o, name=n: record_activations(name, o))
                self.hooks.append(h)
    
    def compute(self, batch: torch.Tensor, c: Union[int, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes `Gradcam` maps.

        Args:
            batch (torch.Tensor):
            c (Union[init, str]): target class index in the output layer. c='max' only for classifier layer, select max activation index for each batch item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `Gradcam` maps for the input batch and `model`'s output for the input batch.
        """
        
        self.register_hooks()
        output = self.model(batch)
        for hook in self.hooks:
            hook.remove()
        
        # Notation follows Grad-CAM article
        y = None
        if c != 'max':
            y = self.activations[self.y_name][:, c]
        else:
            _, max_idx = torch.max(output, dim=1)
            y = self.activations[self.y_name].gather(1, max_idx.view(-1, 1))
        A = self.activations[self.A_name]
        
        alpha = torch.autograd.grad(y, A, torch.ones_like(y))[0]  # eq. 1
        alpha = alpha.mean(-1).mean(-1)  # global average pooling
        
        # eq. 2
        L_gradcam = F.relu((alpha.unsqueeze(-1).unsqueeze(-1) * A).sum(1))
        L_gradcam -= L_gradcam.min()
        return (L_gradcam / L_gradcam.max()).detach(), output.detach()

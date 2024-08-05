#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

"""
EnD: Entangling and Disentangling deep representations for bias correction: 
https://openaccess.thecvf.com/content/CVPR2021/html/Tartaglione_EnD_Entangling_and_Disentangling_Deep_Representations_for_Bias_Correction_CVPR_2021_paper.html
"""

from typing import Union, Optional

import numpy as np
import torch
import torch.nn.functional as F

from EIDOSearch.regularizers.utils import Hook


class Normalize(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(Normalize, self).__init__()
        self.scale = scale
    
    def forward(self, input):
        sizes = input.size()
        if len(sizes) > 2:
            input = input.view(-1, np.prod(sizes[1:]))
            input = torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
            input = input.view(sizes)
        return input


# For each discriminatory class, orthogonalize samples
def end_orthogonal(output: torch.Tensor, gram: torch.Tensor, bias_labels: torch.Tensor) -> torch.Tensor:
    """For each discriminatory class, orthogonalize samples.

    Args:
        output (torch.Tensor):
        gram (torch.Tensor):
        bias_labels (torch.Tensor):

    Returns:
        torch.Tensor: orthogonal loss.
    """
    bias_classes = torch.unique(bias_labels)
    orthogonal_loss = torch.tensor(0.).to(output.device)
    M_tot = 0.
    
    for bias_class in bias_classes:
        bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)
        bias_mask = torch.tril(torch.mm(bias_mask, torch.transpose(bias_mask, 0, 1)), diagonal=-1)
        M = bias_mask.sum()
        M_tot += M
        
        if M > 0:
            orthogonal_loss += torch.sum(torch.abs(gram * bias_mask))
    
    if M_tot > 0:
        orthogonal_loss /= M_tot
    return orthogonal_loss


# For each target class, parallelize samples belonging to
# different discriminatory classes
def end_parallel(gram: torch.Tensor, target_labels: torch.Tensor, bias_labels: torch.Tensor) -> torch.Tensor:
    """For each target class, parallelize samples belonging to different discriminatory classes.

    Args:
        gram (torch.Tensor): Gram matrix.
        target_labels (torch.Tensor): target labels.
        bias_labels (torch.Tensor): bias labels.

    Returns:
        torch.Tensor: parallel loss.

    """
    target_classes = torch.unique(target_labels)
    bias_classes = torch.unique(bias_labels)
    parallel_loss = torch.tensor(0.).to(gram.device)
    M_tot = 0.
    
    for target_class in target_classes:
        class_mask = (target_labels == target_class).type(torch.float).unsqueeze(dim=1)
        
        for idx, bias_class in enumerate(bias_classes):
            bias_mask = (bias_labels == bias_class).type(torch.float).unsqueeze(dim=1)
            
            for other_bias_class in bias_classes[idx:]:
                if other_bias_class == bias_class:
                    continue
                
                other_bias_mask = (bias_labels == other_bias_class).type(torch.float).unsqueeze(dim=1)
                mask = torch.tril(torch.mm(class_mask * bias_mask, torch.transpose(class_mask * other_bias_mask, 0, 1)),
                                  diagonal=-1)
                M = mask.sum()
                M_tot += M
                
                if M > 0:
                    parallel_loss -= torch.sum((1.0 + gram) * mask * 0.5)
    
    if M_tot > 0:
        parallel_loss = 1.0 + (parallel_loss / M_tot)
    
    return parallel_loss


def end_regu(hook: Hook, target_labels: torch.Tensor, bias_labels: torch.Tensor, alpha: Optional[float] = 1.0,
             beta: Optional[float] = 1.0, sum: Optional[bool] = True) -> Union[torch.Tensor, tuple]:
    """Computes EnD regularization term: https://arxiv.org/abs/2103.02023.

    Examples:
        ```python

        import EnD


        model = resnet18()
        model.avgpool = nn.Sequential(
        model.avgpool,
        EnD.Normalize()
        )
        hook = EnD.Hook(model.avgpool, backward=False)

        . . .
        def criterion(outputs, target, bias_labels):
        ce = F.cross_entropy(outputs, target)
        end = EnD.end_regu(hook, target, bias_labels, alpha=0.1, beta=0.1)
        return ce + end
        ```

    Args:
        hook (Hook): forward hook applied on the desired layer.
        target_labels (torch.Tensor): ground truth labels of the batch.
        bias_labels (torch.Tensor): bias labels given for the current batch.
        alpha (float, optional): weight of the disentangling term. Defaults to 1.0.
        beta (float, optional): weight of the entangling. Defaults to 1.0.
        sum (bool, optional): if False, returns the contributions of the two terms separately, otherwise sum. Defaults to True.

    Returns:
        Union[torch.Tensor, tuple]: value of the EnD term.
    """
    D = hook.output
    if len(D.size()) > 2:
        D = D.view(-1, np.prod((D.size()[1:])))
    
    gram_matrix = torch.tril(torch.mm(D, torch.transpose(D, 0, 1)), diagonal=-1)
    # not really needed, just for safety for approximate repr
    gram_matrix = torch.clamp(gram_matrix, -1, 1.)
    
    zero = torch.tensor(0.).to(target_labels.device)
    R_ortho = end_orthogonal(D, gram_matrix, bias_labels) if alpha != 0 else zero
    R_parallel = end_parallel(gram_matrix, target_labels, bias_labels) if beta != 0 else zero
    
    if sum:
        return alpha * R_ortho + beta * R_parallel
    return alpha * R_ortho, beta * R_parallel

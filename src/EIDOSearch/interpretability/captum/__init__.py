#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from captum._utils.models import SkLearnLinearRegression, SkLearnLasso
from captum.attr import IntegratedGradients, GradientShap, Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torchvision import models

from EIDOSearch.datasets.transforms import ImageNet


def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    
    plt.show()


class CaptumInterpretability:
    def __init__(self, model, transforms, cmap=None):
        self.model = model
        self.transforms = transforms
        if cmap is None:
            self.cmap = LinearSegmentedColormap.from_list('custom blue',
                                                          [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')],
                                                          N=256)
        else:
            self.cmap = cmap
    
    def get_pred_label_idx(self, input):
        output = self.model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        return pred_label_idx.squeeze_()
    
    def integrated_gradients(self, input, noise_tunnel, **kwargs):
        input = self.transforms(input).unsqueeze(0).to(next(model.parameters()).device)
        pred_label_idx = self.get_pred_label_idx(input)
        
        transposed_input = np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        integrated_gradients = IntegratedGradients(self.model)
        
        if noise_tunnel:
            noise_tunnel = NoiseTunnel(integrated_gradients)
            
            attributions = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
            transposed_attribution = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
            
            return viz.visualize_image_attr(transposed_attribution, transposed_input, use_pyplot=False,
                                            cmap=self.cmap, **kwargs)
        
        else:
            attributions = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)
            transposed_attribution = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
            
            return viz.visualize_image_attr(transposed_attribution, transposed_input, use_pyplot=False,
                                            cmap=self.cmap, **kwargs)
    
    def gradient_shap(self, input, **kwargs):
        input = self.transforms(input).unsqueeze(0).to(next(model.parameters()).device)
        pred_label_idx = self.get_pred_label_idx(input)
        
        transposed_input = np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        gradient_shap = GradientShap(model)
        rand_img_dist = torch.cat([input * 0, input * 1])
        
        attributions = gradient_shap.attribute(input, n_samples=50, stdevs=0.0001, baselines=rand_img_dist,
                                               target=pred_label_idx)
        transposed_attribution = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        return viz.visualize_image_attr(transposed_attribution, transposed_input, use_pyplot=False, cmap=self.cmap,
                                        **kwargs)
    
    def occlusion(self, input, strides, window_shapes, **kwargs):
        occlusion = Occlusion(model)
        
        input = self.transforms(input).unsqueeze(0).to(next(model.parameters()).device)
        pred_label_idx = self.get_pred_label_idx(input)
        
        transposed_input = np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        attributions = occlusion.attribute(input, strides=strides, target=pred_label_idx,
                                           sliding_window_shapes=window_shapes, baselines=0)
        transposed_attribution = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        return viz.visualize_image_attr(transposed_attribution, transposed_input, use_pyplot=False, cmap=self.cmap,
                                        **kwargs)


if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model = model.eval()
    
    # interp = CaptumInterpretability(model, ImageNet[1])
    #
    # fig, _ = interp.integrated_gradients(Image.open('../dog.png'), False, method='heat_map', show_colorbar=True,
    #                                      sign='positive', outlier_perc=1)
    # show_figure(fig)
    #
    # fig, _ = interp.integrated_gradients(Image.open('../dog.png'), True, method='heat_map', show_colorbar=True,
    #                                      sign='positive')
    # show_figure(fig)
    #
    # fig, _ = interp.gradient_shap(Image.open('../dog.png'), method='heat_map', show_colorbar=True,
    #                               sign='absolute_value')
    # show_figure(fig)
    #
    # fig, _ = interp.occlusion(Image.open('../dog.png'), method='heat_map', show_colorbar=True,
    #                           sign='positive', outlier_perc=2)
    # show_figure(fig)

    input = ImageNet[1](Image.open('../dog.png')).unsqueeze(0).to(next(model.parameters()).device)
    outputs = model(input)
    output_probs = F.softmax(outputs, dim=1).squeeze(0)

    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

    lr_lime = Lime(
        model,
        interpretable_model=SkLearnLasso(alpha=0.08),  # build-in wrapped sklearn Linear Regression
        similarity_func=exp_eucl_distance
    )

    label_idx = output_probs.argmax().unsqueeze(0)
    attrs = lr_lime.attribute(
        input,
        target=label_idx,
        n_samples=40,
        perturbations_per_eval=16,
        show_progress=True
    ).squeeze(0)

    print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())

    viz.visualize_image_attr(
        attrs.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels
        method='heat_map',
        sign='all',
        show_colorbar=True
    )

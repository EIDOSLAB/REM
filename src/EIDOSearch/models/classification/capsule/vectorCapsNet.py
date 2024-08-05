"""
PyTorch implementation of Capsule Networks (master's thesis)

Dynamic Routing Between Capsules: https://arxiv.org/abs/1710.09829
Matrix Capsules with EM Routing: https://openreview.net/pdf?id=HJWLfGWRb

Author: Riccardo Renzulli
University: Università degli Studi di Torino, Department of Computer Science
Company: Addfor S.p.A.
"""

#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capsule import CapsClass2d, CapsPrimary2d
from .utils import conv2d_output_shape

"""
Usage example: python main.py --dataset=mnist --model=VectorCapsNet --output_folder=test99 --trial=trial1 
                              --epochs=100 --primary_num_routing_iterations=1 --lr=0.001 
                              --reconstruction=FCDecoder --batch_size=128
"""


class VectorCapsNet(nn.Module):
    
    def __init__(self, config, device):
        super(VectorCapsNet, self).__init__()
        self.config = config
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=config.input_channels,
                               out_channels=config.out_channels_conv1,
                               kernel_size=config.kernel_size_conv1,
                               stride=config.stride_conv1)
        
        self.h0, self.w0 = conv2d_output_shape((config.input_height, config.input_width),
                                               kernel_size=config.kernel_size_conv1,
                                               stride=config.stride_conv1,
                                               pad=config.padding_conv1,
                                               dilation=config.dilation_conv1)
        
        self.primaryCaps = CapsPrimary2d(input_channels=config.out_channels_conv1,
                                         input_height=self.h0,
                                         input_width=self.w0,
                                         kernel_size=config.kernel_size_primaryCaps,
                                         stride=config.stride_primaryCaps,
                                         padding=config.padding_primaryCaps,
                                         dilation=config.dilation_primaryCaps,
                                         routing_method=config.routing,
                                         num_iterations=config.primary_num_routing_iterations,
                                         squashing=config.squashing_primaryCaps,
                                         output_caps_types=config.num_primaryCaps_types,
                                         output_caps_shape=config.dim_primaryCaps,
                                         device=device)
        
        h1, w1 = conv2d_output_shape((self.h0, self.w0),
                                     kernel_size=config.kernel_size_primaryCaps,
                                     stride=config.stride_primaryCaps,
                                     pad=config.padding_primaryCaps,
                                     dilation=config.dilation_primaryCaps)
        
        self.num_primary_units = h1 * w1 * config.num_primaryCaps_types
        
        self.classCaps = CapsClass2d(input_height=h1,
                                     input_width=w1,
                                     routing_method=config.routing,
                                     num_iterations=config.num_routing_iterations,
                                     input_caps_types=config.num_primaryCaps_types,
                                     input_caps_shape=config.dim_primaryCaps,
                                     output_caps_types=config.num_classes,
                                     output_caps_shape=config.dim_classCaps,
                                     transform_share=config.transform_share_classCaps,
                                     device=device)
        
        if config.reconstruction == "FCDecoder":
            self.decoder = FCDecoder(config,
                                     in_features_fc1=config.dim_classCaps[0] * config.dim_classCaps[
                                         1] * config.num_classes,
                                     out_features_fc1=config.out_features_fc1,
                                     out_features_fc2=config.out_features_fc2,
                                     out_features_fc3=config.out_features_fc3,
                                     device=device)
        elif config.reconstruction == "FCDecoder2":
            self.decoder = FCDecoder2(config,
                                      in_features_fc1=config.dim_classCaps[0] * config.dim_classCaps[
                                          1] * config.num_classes,
                                      out_features_fc0=config.out_features_fc0,
                                      out_features_fc1=config.out_features_fc1,
                                      out_features_fc2=config.out_features_fc2,
                                      out_features_fc3=config.out_features_fc3,
                                      device=device)
        else:
            self.decoder = None
    
    def forward(self, x, target=None):
        """
        The dimension transformation procedure of an input tensor in each layer:
            0. input: [b, c0, h0, w0] -->
            1. conv1 --> [b, c1, h1, w1] -->
            2. primaryCaps poses --> [b, B, h2, w2, is0, is1] -->
            3. classCaps poses --> [b, C, 1, 1, os0, os1] -->
            4. view poses --> [b, C, os0, os1]

        :param x: Image tensor, shape [b, channels, ih, iw]
        :param target: One-hot encoded target tensor, shape [b, num_classes]
        :return: (output_caps_poses, output_caps_activations)
                 The capsules poses and activations tensors of capsule class layer.
                 output_caps_poses: [b, C, os0, os1], output_caps_activations: [b, C]
        """
        batch_size = x.size(0)
        
        # Input: [b, c0, h0, w0]
        x = F.relu(self.conv1(x))
        
        # x: [b, c1, h1, w1]
        output_caps_poses, output_caps_activations = self.primaryCaps(x)
        self.primary_caps_activations = output_caps_activations
        # output_caps_poses: [b, B, h2, w2, is0, is1]
        # output_caps_activations: [b, B, h2, w2]
        if self.config.coupl_coeff:
            output_caps_poses, output_caps_activations, coupling_coefficients = self.classCaps(output_caps_poses,
                                                                                               output_caps_activations,
                                                                                               True)
        else:
            output_caps_poses, output_caps_activations = self.classCaps(output_caps_poses, output_caps_activations)
        # output_caps_poses: [b, C, 1, 1, os0, os1]
        # output_caps_activations: [b, C, 1, 1]
        
        if self.decoder is not None:
            reconstructions = self.decoder(output_caps_poses, output_caps_activations, target)
            # reconstructions: [b, c0 * h0 * w0]
            output_caps_poses = output_caps_poses.view(batch_size, output_caps_poses.size(1),
                                                       output_caps_poses.size(-2),
                                                       output_caps_poses.size(-1))
            output_caps_activations = output_caps_activations.view(batch_size, output_caps_poses.size(1))
            # output_caps_poses: [b, C, os0, os1]
            # output_caps_activations: [b, C]
            if self.config.coupl_coeff:
                return output_caps_poses, output_caps_activations, coupling_coefficients, reconstructions
            else:
                return output_caps_poses, output_caps_activations, reconstructions
        
        else:
            output_caps_poses = output_caps_poses.view(batch_size, output_caps_poses.size(1),
                                                       output_caps_poses.size(-2),
                                                       output_caps_poses.size(-1))
            output_caps_activations = output_caps_activations.view(batch_size, output_caps_poses.size(1))
            # output_caps_poses: [b, C, os0, os1]
            # output_caps_activations: [b, C]
            if self.config.coupl_coeff:
                return output_caps_poses, output_caps_activations, coupling_coefficients
            else:
                return output_caps_poses, output_caps_activations


class FCDecoder(nn.Module):
    
    def __init__(self, config, in_features_fc1, out_features_fc1, out_features_fc2, out_features_fc3, device):
        """
        A fully-connected feed-forward decoder network.

        :param in_features_fc1: FC1 input features.
        :param out_features_fc1: FC1 output features.
        :param out_features_fc2: FC2 input features.
        :param out_features_fc3: FC2 output features.
        :param device: cpu or gpu tensor.
        """
        super(FCDecoder, self).__init__()
        self.config = config
        self.device = device
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        self.fc2 = nn.Linear(out_features_fc1, out_features_fc2)
        self.fc3 = nn.Linear(out_features_fc2, out_features_fc3)
    
    def forward(self, input_caps_poses, input_caps_activations, target=None):
        """
        :param input_caps_poses: Class capsules poses, shape [b, num_classes, is0, is1]
        :param input_caps_activations: Class capsules activations, shape [b, num_classes]
        :param target: One-hot encoded target tensor, shape[b, num_classes]
        :return: reconstructions: The reconstructions of original images, shape [b, c0, h0, w0]
        """
        batch_size = input_caps_poses.size(0)
        input_caps_types = input_caps_poses.size(1)
        input_caps_shape = (input_caps_poses.size(-2), input_caps_poses.size(-1))
        input_caps_poses = input_caps_poses.view(batch_size,
                                                 input_caps_types,
                                                 input_caps_shape[0] * input_caps_shape[1])
        
        if target is None:
            norms = torch.sqrt(torch.sum(input_caps_poses ** 2, dim=2))
            pred = norms.max(1, keepdim=True)[1].type(torch.LongTensor)
            target = F.one_hot(pred.view(-1, 1), input_caps_poses.size(1))
        
        target = target.type(torch.FloatTensor).to(self.device)
        
        mask = target.permute(0, 2, 1)  # mask: [b, num_classes, 1]
        # print(target.size())
        # print(input_caps_poses.size())
        input_caps_poses_masked = mask * input_caps_poses  # input_caps_poses_masked: [b, num_classes, is0, is1]
        input_caps_poses_masked = input_caps_poses_masked.view(batch_size, -1)
        # input_caps_poses_masked: [b, num_classes * is0 * is1]
        
        input_caps_poses = F.relu(self.fc1(input_caps_poses_masked))
        
        input_caps_poses = F.relu(self.fc2(input_caps_poses))
        
        reconstructions = torch.sigmoid(self.fc3(input_caps_poses))
        # reconstructions: [b, c0 * h0 * w0]
        reconstructions = reconstructions.view(batch_size, self.config.input_channels,
                                               self.config.input_height, self.config.input_width)
        # reconstructions: [b, c0, h0, w0]
        return reconstructions


class FCDecoder2(nn.Module):
    
    def __init__(self, config, in_features_fc1, out_features_fc0, out_features_fc1, out_features_fc2, out_features_fc3,
                 device):
        """
        A fully-connected feed-forward decoder network.

        :param in_features_fc1: FC1 input features.
        :param out_features_fc0: FC0 output features.
        :param out_features_fc1: FC1 output features.
        :param out_features_fc2: FC2 input features.
        :param out_features_fc3: FC2 output features.
        :param device: cpu or gpu tensor.
        """
        super(FCDecoder2, self).__init__()
        self.config = config
        self.device = device
        self.fc0 = nn.Linear(in_features_fc1, out_features_fc0)
        self.fc1 = nn.Linear(out_features_fc0, out_features_fc1)
        self.fc2 = nn.Linear(out_features_fc1, out_features_fc2)
        self.fc3 = nn.Linear(out_features_fc2, out_features_fc3)
    
    def forward(self, input_caps_poses, input_caps_activations, target=None):
        """
        :param input_caps_poses: Class capsules poses, shape [b, num_classes, is0, is1]
        :param input_caps_activations: Class capsules activations, shape [b, num_classes]
        :param target: One-hot encoded target tensor, shape[b, num_classes]
        :return: reconstructions: The reconstructions of original images, shape [b, c0, h0, w0]
        """
        batch_size = input_caps_poses.size(0)
        input_caps_types = input_caps_poses.size(1)
        input_caps_shape = (input_caps_poses.size(-2), input_caps_poses.size(-1))
        input_caps_poses = input_caps_poses.view(batch_size,
                                                 input_caps_types,
                                                 input_caps_shape[0] * input_caps_shape[1])
        
        if target is None:
            norms = torch.sqrt(torch.sum(input_caps_poses ** 2, dim=2))
            pred = norms.max(1, keepdim=True)[1].type(torch.LongTensor)
            target = torch.one_hot((batch_size, input_caps_poses.size(1)), pred.view(-1, 1))
        
        target = target.type(torch.FloatTensor)
        
        mask = target.unsqueeze(2).to(self.device)  # mask: [b, num_classes, 1]
        input_caps_poses_masked = mask * input_caps_poses  # input_caps_poses_masked: [b, num_classes, is0, is1]
        input_caps_poses_masked = input_caps_poses_masked.view(batch_size, -1)
        # input_caps_poses_masked: [b, num_classes * is0 * is1]
        
        input_caps_poses = F.relu(self.fc0(input_caps_poses_masked))
        
        input_caps_poses = F.relu(self.fc1(input_caps_poses))
        
        input_caps_poses = F.relu(self.fc2(input_caps_poses))
        
        reconstructions = torch.sigmoid(self.fc3(input_caps_poses))
        # reconstructions: [b, c0 * h0 * w0]
        reconstructions = reconstructions.view(batch_size, self.config.input_channels,
                                               self.config.input_height, self.config.input_width)
        # reconstructions: [b, c0, h0, w0]
        return reconstructions

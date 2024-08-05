import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.capsule import LinearCaps2d, CapsPrimary2d
from models.vectorCapsNet import FCDecoder
from ops.utils import conv2d_output_shape

class ResNet18VectorCapsNet(nn.Module):

    def __init__(self, config, device, bins=None):
        super(ResNet18VectorCapsNet, self).__init__()
        self.config = config
        self.device = device
        self.bins = bins

        self.resnet18 = torchvision.models.resnet18(pretrained=True)

        modules=list(self.resnet18.children())[:-2]

        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

        self.resnet18.eval()

        self.c0, self.h0, self.w0 = 512, 7, 7

        self.padding = torch.nn.ZeroPad2d(6)

        self.c0, self.h0, self.w0 = 512, 19, 19
        self.primaryCaps = CapsPrimary2d(input_channels=self.c0,
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
        self.h1, self.w1 = h1, w1

        print(h1,w1)

        self.num_primary_units = h1 * w1 * config.num_primaryCaps_types
        self.classCaps = LinearCaps2d(input_height=h1,
                                     input_width=w1,
                                     routing_method=config.routing,
                                     num_iterations=config.num_routing_iterations,
                                     input_caps_types=config.num_primaryCaps_types,
                                     input_caps_shape=config.dim_primaryCaps,
                                     output_caps_types=config.num_classes,
                                     output_caps_shape=config.dim_classCaps,
                                     transform_share=config.transform_share_classCaps,
                                     binning=config.binning,
                                     bins=self.bins,
                                     device=device)


        if config.reconstruction == "FCDecoder":
            self.decoder = FCDecoder(config,
                                     in_features_fc1=config.dim_classCaps[0] * config.dim_classCaps[
                                         1] * config.num_classes,
                                     out_features_fc1=config.out_features_fc1,
                                     out_features_fc2=config.out_features_fc2,
                                     out_features_fc3=config.out_features_fc3,
                                     batch_normalization=config.batch_normalization,
                                     device=device)
        else: self.decoder = None

    def forward(self, x, target=None):
        """
        The dimension transformation procedure of an input tensor in each layer:
            0. input: [b, c0, h0, w0] -->
            1. resnet18 --> [b, c1, h1, w1] -->
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
        x = F.relu(self.resnet18(x))
        #print(x.size())
        x = self.padding(x)
        #print(x.size())
    
        # x: [b, c1, h1, w1]
        output_caps_poses, output_caps_activations = self.primaryCaps(x)
        self.primary_caps_activations = output_caps_activations
        # output_caps_poses: [b, B, h2, w2, is0, is1]
        # output_caps_activations: [b, B, h2, w2]
        if self.config.coupl_coeff:
            output_caps_poses, output_caps_activations, coupling_coefficients = self.classCaps(output_caps_poses, output_caps_activations, True)
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
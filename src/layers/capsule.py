"""
In the documentation I use the following notation to describe tensors shapes

b: batch size
B: number of input capsule types
C: number of output capsule types
ih: input height
iw: input width
oh: output height
ow: output width
is0: first dimension of input capsules
is1: second dimension of input capsules
os0: first dimension of output capsules
os1: second dimension of output capsules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops.caps_utils as caps_ops
import ops.utils as ops
from torch.nn.modules.utils import _pair


class CapsPrimary2d(nn.Module):

    def __init__(self, input_channels, input_height, input_width, kernel_size=3, stride=2, padding=0, dilation=1,
                 routing_method="dynamic", num_iterations=1, squashing="hinton", output_caps_types=32,
                 output_caps_shape=(8, 1), device="cpu"):
        """
        The primary capsules are the lowest level of multi-dimensional entities.
        Vector CapsPrimary can be seen as a Convolution layer with shape_output_caps[0] * shape_output_caps[1] *
        num_caps_types channels with squashing as its block non-linearity.

        :param input_channels: The number of input channels.
        :param input_height: Input height dimension
        :param input_width: Input width dimension
        :param kernel_size: The size of the receptive fields, a single number or a tuple.
        :param stride: The stride with which we slide the filters, a single number or a tuple.
        :param padding: The amount by which the input volume is padded with zeros around the border.
        :param dilation: Controls the spacing between the kernel points.
        :param routing_method: The routing-by-agreement mechanism (dynamic or em).
        :param num_iterations: The number of routing iterations.
        :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                          long vectors get shrunk to a length slightly below 1 (only for vector caps).
        :param output_caps_types: The number of primary caps types (each type is a "block").
        :param output_caps_shape: The shape of the higher-level capsules.
        :param device: cpu or gpu tensor.
        """
        super(CapsPrimary2d, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        output_height, output_width = ops.conv2d_output_shape((input_height, input_width), kernel_size, stride,
                                                              padding, dilation)
        self.output_height = output_height
        self.output_width = output_width
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.squashing = squashing
        self.routing_method = routing_method
        self.num_iterations = num_iterations
        self.output_caps_shape = output_caps_shape
        self.output_caps_types = output_caps_types
        self.device = device

        self.caps_poses = nn.Conv2d(in_channels=input_channels,
                                    out_channels=output_caps_shape[0] * output_caps_shape[1] * output_caps_types,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)
        
        if routing_method == "em":
            self.caps_activations = nn.Conv2d(in_channels=input_channels,
                                              out_channels=output_caps_types,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation)

        if self.num_iterations != 0:
                self.routing_bias = nn.Parameter(torch.zeros(output_caps_types,
                                                             output_height, output_width,
                                                             output_caps_shape[0],
                                                             output_caps_shape[1]) + 0.1)

    def forward(self, x):
        """
        :param x: A traditional convolution tensor, shape [b, channels, ih, iw]
        :return: (output_caps_poses, output_caps_activations)
                 The capsules poses and activations tensors of layer L + 1.
                 output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
        """
        batch_size = x.size(0)

        caps = self.caps_poses(x)  # caps: [b, os0 * os1 * C, oh, ow]
        caps = caps.view(batch_size, self.output_caps_types, self.output_caps_shape[0], self.output_caps_shape[1],
                         self.output_height, self.output_width)  # caps: [b, C, os0, os1, oh, ow]
        caps = caps.permute(0, 1, 4, 5, 2, 3)  # caps: [b, C, oh, ow, os0, os1]

        if self.routing_method in ["dynamic", "sda"]:
            output_caps_poses = caps_ops.squash(caps, self.squashing)
            # output_caps_poses: [b, C, oh, ow, os0, os1]
            output_caps_activations = caps_ops.caps_activations(output_caps_poses)
            # output_caps_activations: [b, C, oh, ow]
        elif self.routing_method == "em":
            output_caps_poses = caps
            # output_caps_poses: [b, C, oh, ow, os0, os1]
            output_caps_activations = torch.sigmoid(self.caps_activations(x))
            # output_caps_activations: [b, C, oh, ow]
        else:
            raise ValueError('The routing algorithm {} is not supported yet'.format(self.routing_method))

        if self.num_iterations != 0:
            votes = output_caps_poses.view(batch_size, self.output_caps_types,
                                           self.output_height, self.output_width, 1, 1, 1,
                                           self.output_caps_shape[0], self.output_caps_shape[1])
            # votes: [b, C, oh, ow, B, kh, kw, is0, is1] = [b, C, oh, ow, 1, 1, 1, os0, os1]

            logits = torch.zeros(batch_size, self.output_caps_types, self.output_height, self.output_width,
                                1, 1, 1)  # logits: [b, C, oh, ow, 1, 1, 1]

            logits = logits.to(self.device)

            output_caps_poses, output_caps_activations = caps_ops.routing(self.routing_method, self.num_iterations, votes,
                                                                    logits, self.routing_bias, output_caps_activations)
            # output_caps_poses: [b, C, oh, ow, os0, os1]
            # output_caps_activations: [b, C, oh, ow]

        return output_caps_poses, output_caps_activations

class LinearCaps2d(nn.Module):

    def __init__(self, input_height, input_width, routing_method="dynamic", num_iterations=3, squashing="hinton",
                 input_caps_types=32, input_caps_shape=(16, 1), output_caps_types=10, output_caps_shape=(16, 1),
                 transform_share=False, binning=False, bins=None, device="cpu"):
        """
        It's a fully connected operation between capsule layers.
        It provides the capability of building deep neural network with capsule layers.

        :param input_height: Input height dimension
        :param input_width: Input width dimension
        :param routing_method: The routing-by-agreement mechanism (dynamic or em).
        :param num_iterations: The number of routing iterations.
        :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                          long vectors get shrunk to a length slightly below 1 (only for vector caps).
        :param input_caps_types: The number of input caps types (each type is a "block").
        :param input_caps_shape: The shape of the low-level capsules.
        :param output_caps_types: The number of output caps types (each type is a "block").
        :param output_caps_shape: The shape of the higher-level capsules.
        :param transform_share: Whether or not to share the transformation matrices across capsule in the same channel
                                (i.e. of the same type)
        :param device: cpu or gpu tensor.
        """
        super(LinearCaps2d, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = 1
        self.output_width = 1
        self.routing_method = routing_method
        self.num_iterations = num_iterations
        self.squashing = squashing
        self.input_caps_types = input_caps_types
        self.input_caps_shape = input_caps_shape
        self.output_caps_types = output_caps_types
        self.output_caps_shape = output_caps_shape
        self.kernel_size = (input_height, input_width)
        self.stride = (1, 1)
        self.transform_share = transform_share
        self.binning = binning
        self.bins = torch.tensor(bins).to(device) if bins is not None else bins
        self.device = device

        if not transform_share:
            self.weight = nn.Parameter(torch.nn.init.normal_(torch.empty(self.input_caps_types,
                                                                         self.kernel_size[0],
                                                                         self.kernel_size[1],
                                                                         self.output_caps_types,
                                                                         output_caps_shape[0],
                                                                         input_caps_shape[0]),
                                                             std=0.1))  # weight: [B, ih, iw, C, os0, is0]

        else:
            self.weight = nn.Parameter(torch.nn.init.normal_(torch.empty(1,
                                                                         1,
                                                                         1,
                                                                         self.output_caps_types,
                                                                         output_caps_shape[0],
                                                                         input_caps_shape[0]),
                                                             std=0.1))  # weight: [1, 1, 1, C, os0, is0]

        if routing_method in ["dynamic", "sda"]:
            self.routing_bias = nn.Parameter(torch.zeros(self.output_caps_types,
                                                         self.output_height,
                                                         self.output_width,
                                                         output_caps_shape[0],
                                                         output_caps_shape[1])
                                             + 0.1)
            # routing_bias: [B, oh, ow, os0, os1]
            self.beta_a = None
            self.beta_u = None
        elif routing_method == "em":
            self.routing_bias = None
            self.beta_a = nn.Parameter(torch.randn(output_caps_types))  # beta_a: [C]
            self.beta_u = nn.Parameter(torch.randn(output_caps_types))  # beta_u: [C]
        elif routing_method == "self-attention":
            self.query = nn.Parameter(torch.nn.init.normal_(torch.empty(output_caps_shape[0]*output_caps_shape[1],
                                                                    output_caps_shape[0]*output_caps_shape[1]),
                                                                    std=0.1))  # weight: [B*ih*iw, is0*is1]
            self.key = nn.Parameter(torch.nn.init.normal_(torch.empty(output_caps_shape[0]*output_caps_shape[1],
                                                                    output_caps_shape[0]*output_caps_shape[1]),
                                                                    std=0.1))  # weight: [B*ih*iw, is0*is1]
            self.value = nn.Parameter(torch.nn.init.normal_(torch.empty(output_caps_shape[0]*output_caps_shape[1],
                                                                    output_caps_shape[0]*output_caps_shape[1]),
                                                                    std=0.1))  # weight: [B*ih*iw, is0*is1]
            self.attention = torch.nn.MultiheadAttention(output_caps_shape[0]*output_caps_shape[1], 4)
        else:
            raise ValueError('The routing algorithm {} is not supported yet'.format(routing_method))


    def forward(self, input_caps_poses, input_caps_activations, coupl_coeff=False):
        """
        :param input_caps_poses: The capsules poses tensor of layer L, shape [b, B, ih, iw, is0, is1]
        :param input_caps_activations: The capsules activations tensor of layer L, shape [b, B, ih, iw]
        :return: (output_caps_poses, output_caps_activations)
                 The capsules poses and activations tensors of layer L + 1.
                 output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
        """
        batch_size = input_caps_poses.size(0)

        if self.transform_share:
            transform_matr = self.weight.expand(self.input_caps_types, self.kernel_size[0], self.kernel_size[1],
                                                self.output_caps_types, self.output_caps_shape[0],
                                                self.input_caps_shape[0])
            transform_matr = transform_matr.contiguous()  # transform_matr: [B, ih, iw, C, os0, is0]
        else:
            transform_matr = self.weight  # transform_matr: [B, ih, iw, C, os0, is0]

        votes = caps_ops.convolution_caps2(input_caps_poses, transform_matr, self.kernel_size, self.stride,
                                           self.output_caps_shape, self.device)
        # votes: [b, C, 1, 1, B, ih, iw, os0, os1]

        #requires_grad_logits = self.routing_method != "sda"
        requires_grad_logits = True
        logits = torch.zeros(batch_size, self.output_caps_types, self.output_height, self.output_width,
                             self.input_caps_types, self.kernel_size[0], self.kernel_size[1], requires_grad=requires_grad_logits)
        logits = logits.to(self.device)

        #logits = torch.tile(self.logits, (batch_size,1,1,1,1,1,1))
        if coupl_coeff:
            output_caps, output_caps_activations, coupling_coefficients = caps_ops.routing(self.routing_method, self.num_iterations, votes, logits,
                                                                                           self.routing_bias, input_caps_activations, self.beta_a, self.beta_u, self.squashing, True, self.binning, self.bins)
            return output_caps, output_caps_activations, coupling_coefficients

        else:
            output_caps, output_caps_activations = caps_ops.routing(self.routing_method, self.num_iterations, votes, logits,
                                                                    self.routing_bias, input_caps_activations, self.squashing)
            return output_caps, output_caps_activations
        # output_caps_poses: [b, C, 1, 1, os0, os1]
        # output_caps_activations: [b, C, 1, 1]

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
        self.mean = torch.tensor(self.config.mean).view(len(self.config.mean),1,1).to(device)
        self.std = torch.tensor(self.config.std).view(len(self.config.std),1,1).to(device)

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
            norms = input_caps_activations
            pred = norms.max(1, keepdim=True)[1].type(torch.LongTensor)
            target = F.one_hot(pred.view(-1, 1), input_caps_poses.size(1))
        else:
            target = target[:,None,:]

        target = target.type(torch.FloatTensor).to(self.device)

        mask = target.permute(0,2,1)  # mask: [b, num_classes, 1]
        #print(target.size())
        #print(input_caps_poses.size())
        input_caps_poses_masked = mask * input_caps_poses  # input_caps_poses_masked: [b, num_classes, is0, is1]
        input_caps_poses_masked = input_caps_poses_masked.view(batch_size, -1)
        # input_caps_poses_masked: [b, num_classes * is0 * is1]

        input_caps_poses = F.relu(self.fc1(input_caps_poses_masked))

        input_caps_poses = F.relu(self.fc2(input_caps_poses))

        reconstructions = torch.sigmoid(self.fc3(input_caps_poses))
        # reconstructions: [b, c0 * h0 * w0]
        reconstructions = reconstructions.view(batch_size, self.config.input_channels,
                                               self.config.input_height, self.config.input_width)
        
        #mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).to(reconstructions.device)
        #std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).to(reconstructions.device)
        reconstructions = (reconstructions - self.mean)/self.std
        #reconstructions = (reconstructions-mean)+std
        # reconstructions: [b, c0, h0, w0]
        return reconstructions


class FCDecoder2(nn.Module):

    def __init__(self, config, in_features_fc1, out_features_fc1, out_features_fc2, out_features_fc3, device):
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
            norms = input_caps_activations
            pred = norms.max(1, keepdim=True)[1].type(torch.LongTensor)
            target = F.one_hot(pred.view(-1, 1), input_caps_poses.size(1))
        else:
            target = target[:,None,:]

        target = target.type(torch.FloatTensor).to(self.device)

        mask = target.permute(0,2,1)  # mask: [b, num_classes, 1]
        input_caps_poses_masked = mask * input_caps_poses  # input_caps_poses_masked: [b, num_classes, is0, is1]

        input_caps_poses_masked = input_caps_poses_masked.view(batch_size, -1)
        # input_caps_poses_masked: [b, num_classes * is0 * is1]
        input_caps_poses_masked = input_caps_poses_masked[input_caps_poses_masked != 0].view(batch_size,input_caps_shape[0] * input_caps_shape[1])
        input_caps_poses = F.relu(self.fc1(input_caps_poses_masked))

        input_caps_poses = F.relu(self.fc2(input_caps_poses))

        reconstructions = torch.sigmoid(self.fc3(input_caps_poses))
        # reconstructions: [b, c0 * h0 * w0]
        reconstructions = reconstructions.view(batch_size, self.config.input_channels,
                                               self.config.input_height, self.config.input_width)
        reconstructions = (reconstructions -0.1307)/0.3081
        # reconstructions: [b, c0, h0, w0]
        return reconstructions

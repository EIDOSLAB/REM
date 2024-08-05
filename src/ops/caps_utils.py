"""
PyTorch implementation of REM: Routing Entropy Minimization for Capsule Networks

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
import math
import numpy as np
import ops.utils as ops
from torch.nn.modules.utils import _pair
from pathlib import Path
import cv2

def routing(routing_method, num_iterations, votes, logits, routing_bias, input_caps_activations,
            beta_a=None, beta_u=None, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
    """
    Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of
    higher-level capsules. When multiple predictions agree, a higher level capsule becomes active.
    To achieve these results we use an iterative routing-by-agreement mechanism.
    :param routing_method: The iterative routing-by-agreement mechanism method: dynamic or EM.
    :param num_iterations: The number of routing iterations.
    :param votes: The votes from lower-level capsules to higher-level capsules,
                  shape [b, C, oh, ow, B, kh, kw, os0, os1].
    :param logits: The coupling coefficients that are determined by the routing process, shape [b, C, oh, ow, B, kh, kw]
    :param routing_bias: The routing biases (only for dynamic routing), shape [B, oh, ow, os0, os1].
    :param input_caps_activations: The capsules activations tensor of layer L, shape [b, B, ih, iw].
    :param beta_a: Parameter (one for each output caps type, only for EM routing), shape [C].
    :param beta_u: Parameter (one for each output caps type, only for EM routing), shape [C].
    :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                      long vectors get shrunk to a length slightly below 1 (only for dynamic routing).
    :return: (output_caps_poses, output_caps_activations)
             The capsules poses and activations tensors of layer L + 1.
             output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
    """
    assert num_iterations > 0

    if routing_method == "dynamic":
        return dynamic_routing(num_iterations, votes, logits, routing_bias, squashing, coupl_coeff,binning, bins)
    elif routing_method == "sda":
        return sda_routing(num_iterations, votes, logits, routing_bias, input_caps_activations, squashing, coupl_coeff,binning, bins)
    elif routing_method == "em":
        final_lambda = 1e-02
        eps = 1e-8
        return em_routing(num_iterations, votes, logits, routing_bias, input_caps_activations, beta_a, beta_u, final_lambda, eps, coupl_coeff)
    elif routing_method == "self-attention":
        return self_attention_routing(votes,)

def dynamic_routing(num_iterations, votes, logits, routing_bias, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
    """
    A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar
    product with the prediction coming from the lower-level capsule.

    :param num_iterations: The number of routing iterations.
    :param votes: The votes from lower-level capsules to higher-level capsules,
                  shape [b, C, oh, ow, B, kh, kw, os0, os1].
    :param logits: The coupling coefficients that are determined by the routing process, shape [b, C, oh, ow, B, kh, kw]
    :param routing_bias: The routing biases (only for dynamic routing), shape [C, oh, ow, os0, os1].
    :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                      long vectors get shrunk to a length slightly below 1 (only for dynamic routing).

    :return: (output_caps_poses, output_caps_activations)
             The capsules poses and activations tensors of layer L + 1.
             output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
    """
    batch_size = votes.size(0)
    output_height = logits.size(2)
    output_width = logits.size(3)
    output_caps_types = logits.size(1)
    input_caps_types = votes.size(4)
    kernel_size = (votes.size(5), votes.size(6))
    input_caps = input_caps_types * votes.size(6) * votes.size(6)
    if binning and bins is not None:
        centers = ((bins[1:] + bins[:-1])/2).float()

    votes_detached = votes.detach()
    # Dynamic routing core
    for it in range(num_iterations):
        coupling_coeff = torch.softmax(logits, dim=1)  # logits: [b, C, oh, ow, B, kh, kw]
        if binning and bins is not None:
            if it == num_iterations-1:
                edges_ind = torch.bucketize(coupling_coeff, bins)
                edges_ind[edges_ind==0] = 1
                coupling_coeff = centers[edges_ind-1]
        if it == num_iterations-1:
            weighted_votes = votes * coupling_coeff[:, :, :, :, :, :, :, None, None]
        else:
            weighted_votes = votes_detached * coupling_coeff[:, :, :, :, :, :, :, None, None]
        # weighted_votes: [b, C, oh, ow, B, kh, kw, os0, os1]
        output_caps_poses = torch.sum(weighted_votes, dim=(4, 5, 6)) + routing_bias
        # output_caps_poses: [b, C, oh, ow, os0, os1]
        output_caps_poses = squash(output_caps_poses, squashing)
        # output_caps_poses: [b, C, oh, ow, os0, os1]

        # similarities: [b, C, oh, ow, B, kh, kw]
        if num_iterations > 1 and it < num_iterations - 1:
            similarities = torch.matmul(output_caps_poses.view(batch_size, output_caps_types, output_height, output_width,
                                                1, 1, 1, 1, -1),
                                        votes_detached.view(batch_size, output_caps_types, output_height, output_width,
                                                input_caps_types, kernel_size[0], kernel_size[1], -1, 1))
            # similarities: [b, C, oh, ow, B, kh, kw, 1, 1]
            similarities = similarities.squeeze(-1).squeeze(-1)
            logits = logits + similarities
            # logits: [b, C, oh, ow, B, kh, kw]

    output_caps_activations = caps_activations(output_caps_poses)

    if coupl_coeff:
        return output_caps_poses, output_caps_activations, coupling_coeff
    else:
        return output_caps_poses, output_caps_activations

#Modified shapes from https://github.com/moejoe95/MoCapsNet/blob/main/capsules.py#L149
# def sda_routing(num_iterations, votes, logits, routing_bias, input_caps_activations, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
#     """
#     Scaled-distance-agreement routing by Peer et al.
#     """
#     # u -> (batch_size, in_caps, in_dim)
#     batch_size = u.size(0)
#     u_norm = torch.norm(u, dim=-1)

#     u = torch.unsqueeze(torch.unsqueeze(u, 1), 3)
#     u = torch.tile(u, [1, self.num_caps, 1, 1, 1])
#     u = torch.tile(u, [1, 1, 1, self.dim_caps, 1])

#     # tile over batch size
#     w = torch.tile(self.W, [batch_size, 1, 1, 1, 1])
#     # Dotwise product between u and w to get all votes
#     u_hat = torch.sum(u * w, dim=-1)
#     # ensure that ||u_hat|| <= ||v_i||
#     u_hat = self.restrict_prediction(u_hat, u_norm)
#     bias = torch.tile(self.bias, [batch_size, 1, 1])

#     # prevent gradient flow while routing
#     torch.set_grad_enabled(False)

#     b_ij = torch.zeros(batch_size, self.num_caps,
#                         self.in_caps, 1, requires_grad=False).to(self.device)
#     for r in range(num_iterations):
#         c_ij = F.softmax(b_ij, dim=1)
#         c_ij_tiled = torch.tile(c_ij, [1, 1, 1, self.dim_caps])
#         if r == num_iterations - 1:
#             # enable gradient flow in last iteration
#             torch.set_grad_enabled(True)
#         s_j = torch.sum(c_ij_tiled * u_hat, dim=2) + bias
#         v_j = squash(s_j)
#         if r < num_iterations - 1:
#             v_j = torch.unsqueeze(v_j, 2)
#             v_j = torch.tile(v_j, [1, 1, self.in_caps, 1])

#             # calculate scale factor t
#             p_p = 0.9
#             d = torch.norm(v_j - u_hat, dim=-1, keepdim=True)
#             d_o = torch.mean(torch.mean(d)).item()
#             d_p = d_o * 0.5
#             t = np.log(p_p * (self.num_caps - 1)) - \
#                 np.log(1 - p_p) / (d_p - d_o + 1e-12)

#             b_ij = t * d

#     return v_j

def sda_routing(num_iterations, votes, logits, routing_bias, input_caps_activations, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
    """
    Scaled-distance-agreement routing by Peer et al.

    :param num_iterations: The number of routing iterations.
    :param votes: The votes from lower-level capsules to higher-level capsules,
                  shape [b, C, oh, ow, B, kh, kw, os0, os1].
    :param logits: The coupling coefficients that are determined by the routing process, shape [b, C, oh, ow, B, kh, kw]
    :param routing_bias: The routing biases (only for dynamic routing), shape [C, oh, ow, os0, os1].
    :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                      long vectors get shrunk to a length slightly below 1 (only for dynamic routing).

    :return: (output_caps_poses, output_caps_activations)
             The capsules poses and activations tensors of layer L + 1.
             output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
    """
    output_caps_types = logits.size(1)
    if binning and bins is not None:
        centers = ((bins[1:] + bins[:-1])/2).float()

    input_caps_activations = input_caps_activations[:,None,None,None,:,:,:]
    # ensure that ||u_hat|| <= ||v_i||
    votes_activations = votes.norm(dim=(-2,-1))
    votes_normalized = votes/votes.norm(dim=(-2,-1), keepdim=True)    
    votes = (torch.minimum(input_caps_activations, votes_activations)[:,:,:,:,:,:,:,None,None])*votes_normalized

    # prevent gradient flow while routing
    torch.set_grad_enabled(False)
    
    # Scaled-distance-agreement (SDA) routing algorithm core    
    for it in range(num_iterations):
        coupling_coeff = torch.softmax(logits, dim=1)  # logits: [b, C, oh, ow, B, kh, kw]
        if binning and bins is not None:
            if it == num_iterations-1:
                edges_ind = torch.bucketize(coupling_coeff, bins)
                edges_ind[edges_ind==0] = 1
                coupling_coeff = centers[edges_ind-1]
        if it == num_iterations-1:
            torch.set_grad_enabled(True)
        weighted_votes = votes * coupling_coeff[:, :, :, :, :, :, :, None, None]
        # weighted_votes: [b, C, oh, ow, B, kh, kw, os0, os1]
        output_caps_poses = torch.sum(weighted_votes, dim=(4, 5, 6)) + routing_bias
        # output_caps_poses: [b, C, oh, ow, os0, os1]
        output_caps_poses = squash(output_caps_poses, squashing)
        # output_caps_poses: [b, C, oh, ow, os0, os1]

        # similarities: [b, C, oh, ow, B, kh, kw]
        if num_iterations > 1 and it < num_iterations - 1:

                # (batch_size, out_caps, in_caps, out_dim)
                
                # Calculate scale factor t
                p_p = 0.9
                output_caps_poses = output_caps_poses[:,:,:,:,None,None,None,:,:]
                d = torch.norm(output_caps_poses-votes, dim=(-2,-1), keepdim=True)
                d_o = torch.mean(torch.mean(d)).item()
                d_p = d_o * 0.5
                t = (np.log(p_p*(output_caps_types-1)) - np.log(1-p_p))/(d_p - d_o + 1e-12)

                # Calc log prior using inverse distances
                logits = (t * d).squeeze(-1).squeeze(-1)
                # logits: [b, C, oh, ow, B, kh, kw]

    output_caps_activations = caps_activations(output_caps_poses)

    if coupl_coeff:
        return output_caps_poses, output_caps_activations, coupling_coeff
    else:
        return output_caps_poses, output_caps_activations

# def self_attention_routing(votes, query, key, value):
#     query = torch.matmul(input_caps_poses, self.query)
#     key = torch.matmul(input_caps_poses, self.key)
#     value = torch.matmul(input_caps_poses, self.value)
#     input_caps_poses, input_caps_poses_att_weights = self.attention(query, key, value)

def m_step(a_in, r, v, eps, b, B, C, psize, beta_a, beta_u, temp):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))
            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = torch.sigmoid(temp*(beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

def e_step(mu, sigma_sq, a_out, v, eps, b, C):
    """
        ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
        r = softmax(ln(a_j*p_j))
            = softmax(ln(a_j) + ln(p_j))
        Input:
            mu:        (b, 1, C, P*P)
            sigma:     (b, 1, C, P*P)
            a_out:     (b, C, 1)
            v:         (b, B, C, P*P)
        Local:
            ln_p_j_h:  (b, B, C, P*P)
            ln_ap:     (b, B, C, 1)
        Output:
            r:         (b, B, C, 1)
    """
    ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
    ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                - torch.log(sigma_sq.sqrt()) \
                - 0.5*ln_2pi

    ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
    r = torch.softmax(ln_ap, dim=2)
    return r

def em_routing(num_iterations, votes, logits, routing_bias, input_caps_activations, beta_a, beta_u, final_lambda, eps, coupl_coeff=False):
    """
        Input:
            v:         (b, B, C, P*P)
            a_in:      (b, C, 1)
        Output:
            mu:        (b, 1, C, P*P)
            a_out:     (b, C, 1)
        Note that some dimensions are merged
        for computation convenient, that is
        `b == batch_size*oh*ow`,
        `B == self.K*self.K*self.B`,
        `psize == self.P*self.P`
    """

    batch_size = votes.size(0)
    output_height = logits.size(2)
    output_width = logits.size(3)
    output_caps_types = logits.size(1)
    input_caps_types = votes.size(4)
    kernel_size = (votes.size(5), votes.size(6))
    output_shape = (votes.size(7), votes.size(8))

    votes = votes.reshape(batch_size, votes.size(4) * votes.size(5) * votes.size(6), votes.size(1), votes.size(7) * votes.size(8))
    input_caps_activations = input_caps_activations.view(batch_size, -1, 1)

    batch_size, B, c, psize = votes.shape 
    assert c == output_caps_types
    assert (batch_size, B, 1) == input_caps_activations.shape

    r = torch.cuda.FloatTensor(batch_size, B, output_caps_types)
    r = r.detach()
    r = r.fill_(1./output_caps_types)
    for iter_ in range(num_iterations):
        lambda_iter = final_lambda * (1. - torch.pow(torch.tensor(0.95), torch.tensor(1)))
        #lambda_iter = final_lambda * (1. - torch.pow(torch.tensor(0.95), torch.tensor(iter_+1)))
        #lambda_iter = 0.00001
        a_out, mu, sigma_sq = m_step(input_caps_activations, r, votes, eps, batch_size, B, output_caps_types, psize, beta_a, beta_u, lambda_iter)
        if iter_ < num_iterations - 1:
            r = e_step(mu, sigma_sq, a_out, votes, eps, batch_size, output_caps_types)

    #output_caps_poses: [b, C, oh, ow, os0, os1]
    output_caps_activations = a_out.view(batch_size, output_caps_types, output_height, output_width)
    output_caps_poses = mu.view(batch_size, output_caps_types, output_height, output_width, output_shape[0], output_shape[1])
    # logits: [b, C, oh, ow, B, kh, kw]
    coupling_coeff = r.view(batch_size, output_caps_types, output_height, output_width, input_caps_types, kernel_size[0], kernel_size[1]).detach()

    if coupl_coeff:
        return output_caps_poses, output_caps_activations, coupling_coeff
    else:
        return output_caps_poses, output_caps_activations
    return mu, a_out

def add_pathes(self, x, B, K, psize, stride):
    """
        Shape:
            Input:     (b, H, W, B*(P*P+1))
            Output:    (b, H', W', K, K, B*(P*P+1))
    """
    b, h, w, c = x.shape
    assert h == w
    assert c == B*(psize+1)
    oh = ow = int(((h - K )/stride)+ 1) # moein - changed from: oh = ow = int((h - K + 1) / stride)
    idxs = [[(h_idx + k_idx) \
            for k_idx in range(0, K)] \
            for h_idx in range(0, h - K + 1, stride)]
    x = x[:, idxs, :, :]
    x = x[:, :, :, idxs, :]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, oh, ow

def transform_view(self, x, w, C, P, w_shared=False):
    """
        For conv_caps:
            Input:     (b*H*W, K*K*B, P*P)
            Output:    (b*H*W, K*K*B, C, P*P)
        For class_caps:
            Input:     (b, H*W*B, P*P)
            Output:    (b, H*W*B, C, P*P)
    """
    b, B, psize = x.shape
    assert psize == P*P

    x = x.view(b, B, 1, P, P)
    if w_shared:
        hw = int(B / w.size(1))
        w = w.repeat(1, hw, 1, 1, 1)

    w = w.repeat(b, 1, 1, 1, 1)
    x = x.repeat(1, 1, C, 1, 1)
    v = torch.matmul(x, w)
    v = v.view(b, B, C, P*P)
    return v

def add_coord(self, v, b, h, w, B, C, psize):
    """
        Shape:
            Input:     (b, H*W*B, C, P*P)
            Output:    (b, H*W*B, C, P*P)
    """
    assert h == w
    v = v.view(b, h, w, B, C, psize)
    coor = torch.arange(h, dtype=torch.float32) / h
    coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
    coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
    coor_h[0, :, 0, 0, 0, 0] = coor
    coor_w[0, 0, :, 0, 0, 1] = coor
    v = v + coor_h + coor_w
    v = v.view(b, h*w*B, C, psize)
    return v

def caps_activations(caps_poses):
      return caps_poses.norm(dim=(-2,-1))

def squash(caps_poses, squashing_type):
    """
    The non-linear function to ensure that short vectors get shrunk to almost zero length and
    long vectors get shrunk to a length slightly below 1 (only for dynamic routing).

    :param caps_poses: The capsules poses, shape [b, B, ih, iw, is0, is1]
    :param squashing_type: The squashing type

    :return: The capsules poses squashed, shape [b, B, ih, iw, is0, is1]
    """
    if squashing_type == "hinton":
        squared_norm = torch.sum(caps_poses ** 2, dim=(-1, -2), keepdim=True)
        norm = torch.sqrt(squared_norm+1e-6)
        #print(torch.any(norm.isnan()))
        scale = squared_norm / (1 + squared_norm)
        caps_poses = scale * caps_poses / norm
        return caps_poses

def normalized_reconstruction_loss_lambda(config, u, dev_st):
    assert len(u) == len(dev_st)

    u = np.array(list(u))
    dev_st = np.array(list(dev_st))

    original_min_input = np.zeros(u.shape)
    original_max_input = np.ones(u.shape)

    normalized_min_input = (original_min_input - u) / dev_st
    normalized_max_input = (original_max_input - u) / dev_st

    tot_pixels = config.input_channels * config.input_height * config.input_width
    original_max_loss = config.reconstruction_loss_lambda * np.sum(
        (original_min_input - original_max_input) ** 2) * tot_pixels
    normalized_max_loss = np.sum((normalized_min_input - normalized_max_input) ** 2) * tot_pixels

    normalized_lambda = original_max_loss / normalized_max_loss
    return normalized_lambda.item()

def convolution_caps2(input_caps_poses, transform_matr, kernel_size, stride, output_caps_shape, device):
    """
    The convolution operation between capsule layers useful to compute the votes.
    :param input_caps_poses: The input capsules poses, shape [b, B, ih, iw, is0, is1]
    :param transform_matr: The transformation matrices, shape [B, kh, kw, C, os0, is0]
    :param kernel_size: The size of the receptive fields, a single number or a tuple.
    :param stride: The stride with which we slide the filters, a single number or a tuple.
    :param output_caps_shape: The shape of the higher-level capsules.
    :param device: cpu or gpu tensor.
    :return: The votes from lower-level capsules to higher-level capsules, shape [b, C, oh, ow, B, kh, kw, os0, os1].
    """
    batch_size = input_caps_poses.size(0)
    input_caps_types = input_caps_poses.size(1)
    input_height = input_caps_poses.size(2)
    input_width = input_caps_poses.size(3)
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    output_height, output_width = ops.conv2d_output_shape((input_height, input_width), kernel_size, stride)
    input_caps_shape = (input_caps_poses.size(-2), input_caps_poses.size(-1))

    # used to store every capsule i's poses in each capsule c's receptive field
    poses = torch.stack([input_caps_poses[:, :, stride[0] * i:stride[0] * i + kernel_size[0], stride[1] * j:stride[1] * j + kernel_size[1], :, :]
                         for i in range(output_height) for j in range(output_width)], dim=-1)
    # poses: [b, B, kh, kw, is0, is1, oh * ow]
    poses = poses.permute(0, 1, 2, 3, 6, 4, 5)
    # poses: [b, B, kh, kw, oh * ow, is0, is1]
    poses = poses.view(batch_size, input_caps_types, kernel_size[0], kernel_size[1],
                       1, output_height, output_width, input_caps_shape[0], input_caps_shape[1])
    # poses: [b, B, kh, kw, 1, oh, ow, is0, is1]

    transform_matr = transform_matr[None, :, :, :, :, None, None, :, :]
    # transform_matr: [1, B, kh, kw, C, 1, 1, os0, is0]
    votes = torch.matmul(transform_matr, poses)
    # votes: [b, B, kh, kw, C, oh, ow, os0, os1] is1 and os1 should be equals
    return votes.permute(0, 4, 5, 6, 1, 2, 3, 7, 8)  # votes: [b, C, oh, ow, B, kh, kw, os0, os1]

def save_dict_images(config, coupling_coefficients, data, pred, label, class_idx, bins, dataset, output_folder):
    pred = pred.view(label.size()[0], label.size()[1])
    batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    coupling_coefficients = coupling_coefficients.view(batch_size, C*k*k, B*h*w)
    indices = torch.where(pred==class_idx)[1]
    np_cij = coupling_coefficients[indices][:,class_idx,:].cpu().detach().numpy()
    np_data = data[indices].cpu().detach().numpy()
    bin_inds = np.digitize(np_cij, bins, right=False)
    bin_inds_str = []
    for i in range(bin_inds.shape[0]):
        bin_inds_str.append(''.join(str(e) for e in bin_inds[i]))

    for i, string in enumerate(bin_inds_str):
        rle = ops.formatOutput(ops.encode(string))
        dir_path = "{}/{}/{}/originals/{}/{}/".format(output_folder, dataset, config.split, class_idx, rle)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        img_path = "{}/{}/{}/originals/{}/{}/{}.jpg".format(output_folder, dataset, config.split, class_idx, rle, i)
        np_img = np_data[i,0,:,:]
        np_img = (np_img*config.std + config.mean)*255
        cv2.imwrite(img_path, np_img)


def compute_dict_coupl_coeff(dict_cij, coupling_coefficients, pred, label, class_idx, bins):
    pred = pred.view(label.size()[0], label.size()[1])
    if len(coupling_coefficients.size()) < 7:
        coupling_coefficients = torch.permute(coupling_coefficients, (0, 2, 1)) #so that [batch, C, B]
        coupling_coefficients = coupling_coefficients[:,:,None,None,:, None, None]
    batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    coupling_coefficients = coupling_coefficients.view(batch_size, C*k*k, B*h*w)
    #indices = torch.tensor(np.setdiff1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
    #indices = torch.tensor(np.intersect1d(torch.where(label==class_idx)[1], torch.where(pred==class_idx)[1])).to(pred.device)
    indices = torch.where(pred==class_idx)[1]
    np_cij = coupling_coefficients[indices][:,class_idx,:].cpu().detach().numpy()
    bin_inds = np.digitize(np_cij, bins, right=False)
    bin_inds_str = []
    for i in range(bin_inds.shape[0]):
        bin_inds_str.append(''.join(str(e)+'-' for e in bin_inds[i]))

    for string in bin_inds_str:
        if string in dict_cij:
            dict_cij[string] += 1
        else:
            dict_cij[string] = 1

def compute_entropy_coupl_coeff(coupling_coefficients, pred, label):
    pred = pred.view(label.size()[0], label.size()[1])
    batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    coupling_coefficients = coupling_coefficients.view(batch_size, C*k*k, B*h*w)
    # for cifar10
    result = []
    for i in range(C):
        #indices = torch.tensor(np.setdiff1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        indices = torch.tensor(torch.where(pred==i)[1]).to(pred.device)
        mean_cij = torch.mean(coupling_coefficients[indices][:], dim=0)
        result.append(mean_cij) #num_clx1152
    return torch.stack(result)

def get_prob_cij_k(tot_examples, bin_centers, coupling_coefficients, pred, label):
    pred = pred.view(label.size()[0], label.size()[1])
    batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    prob_cij_k = torch.zeros((len(bin_centers), C*k*k, B*h*w)).to(coupling_coefficients.device)
    coupling_coefficients = coupling_coefficients.view(batch_size, C*k*k, B*h*w)
    # for cifar10
    result = []
    for i in range(C):
        #indices = torch.tensor(np.setdiff1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        #indices = torch.tensor(np.intersect1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        indices = torch.tensor(torch.where(pred==i)[1]).to(pred.device)
        correct_cij = coupling_coefficients[indices][:] # bx10x1152
        for c, center in enumerate(bin_centers):
            #prob_cij_k[c] += correct_cij[correct_cij==c].view(-1).size(0) / tot_examples
            prob_cij_k[c] += torch.sum(correct_cij==center, dim=0) / tot_examples
    return prob_cij_k

def estimation_coupl_coeff(coupling_coefficients, pred, label):
    #pred = pred.view(label.size()[0], label.size()[1])
    #batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    #print(coupling_coefficients.max())
    #print(coupling_coefficients.min())

    pred = pred.view(label.size()[0], label.size()[1])
    batch_size, C, k, k, B, h, w = coupling_coefficients.size()
    coupling_coefficients = coupling_coefficients.view(batch_size, C*k*k, B*h*w)
    # for cifar10
    result = []
    for i in range(C):
        #indices = torch.tensor(np.setdiff1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        #indices = torch.tensor(np.intersect1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        indices = torch.tensor(torch.where(pred==i)[1]).to(pred.device)
        mean_cij = torch.mean(coupling_coefficients[indices][:], dim=0)
        result.append(mean_cij[i]) #num_clx1152
    estimated_coupl_coeff = torch.stack(result).view(C, k, k, B, h, w)
    return estimated_coupl_coeff

def compute_entropy_caps(class_caps_poses, pred, label):
    pred = pred.view(label.size()[0], label.size()[1])
    batch_size, C, s0, s1 = class_caps_poses.size()
    class_caps_poses = class_caps_poses.view(batch_size, C, s0*s1)
    # for cifar10
    result = {}
    for i in range(C):
        #indices = torch.tensor(np.setdiff1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        #indices = torch.tensor(np.intersect1d(torch.where(label==i)[1], torch.where(pred==i)[1])).to(pred.device)
        indices = torch.tensor(torch.where(pred==i)[1]).to(pred.device)
        #print((class_caps_poses[indices][:])[i].size())
        result[i] = class_caps_poses[indices,i,:].cpu().detach()
    return result

#https://github.com/peerdavid/gamma-capsule-network/blob/master/capsule/gamma_metrics.py#L5
def get_t_score(coupling_coefficients, device, pred=None, eps=10e-12):
    """
    Compute the t-score of the coupling coefficients in the input mini-batch. 
    It is the normalized average entropy as defined in https://arxiv.org/pdf/1812.09707.pdf
    :param coupling_coefficients: Coupling coefficients between lower-level capsules and higher-level capsules,
                                  shape [b, C, oh, ow, B, ih, iw]. C=output capsule types, B=input capsule types
                                  For example: [128, 10, 1, 1, 32, 6, 6]
    :return: t-score metric
    """
    #if pred is not None:
    #    pred = pred.view(-1)
    #    coupling_coefficients = coupling_coefficients[torch.arange(pred.shape[0]), pred].unsqueeze(1)
    b, C, oh, ow, B, ih, iw = coupling_coefficients.size()
    coupling_coefficients = coupling_coefficients.view(b, C*oh*ow, B*ih*iw)
    coupling_coefficients = coupling_coefficients.permute(0, 2, 1)
    num_input_capsules = coupling_coefficients.size()[1]
    num_output_capsules = coupling_coefficients.size()[2]
    entropy = -torch.sum(coupling_coefficients * torch.log(coupling_coefficients+eps), dim=-1)
    T = 1 - entropy / -torch.log(1 / torch.FloatTensor([num_output_capsules]).to(coupling_coefficients.device))
    #T = 1 - entropy 
    return torch.mean(T).detach().cpu().numpy()
    

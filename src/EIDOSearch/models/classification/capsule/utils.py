#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import fnmatch
import glob
import json
import logging
import os
import random
from copy import deepcopy
from math import floor

import numpy as np
import torch
from torch.nn.modules.utils import _pair


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing spatial output size of conv2d operation.
    It takes a tuple of (h,w) and returns a tuple of (h,w)

    Source: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    pad = _pair(pad)
    dilation = _pair(dilation)
    
    h = floor(((h_w[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    w = floor(((h_w[1] + (2 * pad[0]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return h, w


def save_checkpoint(state, is_best, dir, filename="checkpoint", local=False):
    if local:
        complete_path = dir + "/local_model_best_" + filename + ".pth.tar"
    else:
        complete_path = dir + "/model_best_" + filename + ".pth.tar"
    if is_best:
        for file in os.listdir(dir):
            if local and fnmatch.fnmatch(file, "local_model_best_epoch_*-it{}*".format(state["routing_iterations"])):
                os.remove(dir + "/" + file)
                break
            elif not local and fnmatch.fnmatch(file, "model_best*"):
                os.remove(dir + "/" + file)
                break
        torch.save(state, complete_path)
        # shutil.copyfile(dir + "/" + filename + ".pth.tar", dir + "/model_best_" + filename + ".pth.tar")
    else:
        torch.save(state, dir + "/" + filename + ".pth.tar")


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file "log_path".

    Example: logging.info("Starting training...")

    Source: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    :param log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:

    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params

    Source: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    """
    
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def summary(model, config, model_name="VectorCapsNet"):
    logging.info("=================================================================")
    logging.info("Model architectures: ")
    logging.info(model)
    
    logging.info("Sizes of parameters: ")
    for name, param in model.named_parameters():
        logging.info("{}: {}".format(name, list(param.size())))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.n_params = n_params
    logging.info("=================================================================")
    
    logging.info("----------------------------------------------------------------")
    
    if model_name == "VectorCapsNet":
        non_trainable_params_primary_caps = config.batch_size * model.num_primary_units if config.primary_num_routing_iterations != 0 else 0
        non_trainable_params_class_caps = config.batch_size * model.num_primary_units * config.num_classes
        non_trainable_params = non_trainable_params_primary_caps + non_trainable_params_class_caps
        
        logging.info("Total params: %d " % (n_params + non_trainable_params))
        logging.info("Trainable params: %d " % n_params)
        logging.info("Non-trainable params (coupling coefficients for mini-batch) %d " % non_trainable_params)
    else:
        logging.info("Trainable params: %d " % n_params)
    logging.info("----------------------------------------------------------------")


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __deepcopy__(self, memo=None):
        return DotDict(deepcopy(dict(self), memo=memo))


def save_args(args, dir):
    dict_args = vars(args)
    print(args)
    print(dict_args)
    with open(dir + "/params.json", "w") as outfile:
        json.dump(args, outfile, indent=4)


def get_model_best_path(path, checkpoint="model_best*.pth.tar"):
    for file in glob.glob(os.path.join(path, checkpoint)):
        return file


def get_local_path(path, it):
    checkpoint = "local_model_best_epoch_*-it{}-*.pth.tar".format(it)
    for file in glob.glob(os.path.join(path, checkpoint)):
        return file


def formatnumbers(x):
    x = str(x).replace('.', ',')
    return x


def create_experiment_folder(config):
    if config.routing_annealing:
        training_mode = "annealing"
    else:
        training_mode = "fixed"
    if config.freeze:
        freeze = "freeze"
    else:
        freeze = "nofreeze"
    if config.decay_steps > 0:
        decay = "decay"
    else:
        decay = "nodecay"
    return os.path.join(config.experiment_name, training_mode, freeze, config.monitor, config.caps_loss,
                        config.optimizer, str(config.lr), decay)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

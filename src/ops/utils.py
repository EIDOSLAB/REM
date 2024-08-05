import json
import logging
import os
import fnmatch
import torch
import pathlib
import glob
import wandb
import random
import numpy as np
import torch
from copy import deepcopy
from torch.nn.modules.utils import _pair
from math import floor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def save_checkpoint(state, is_best, dir, filename, epoch, keep_checkpoints=False):
    if keep_checkpoints:
        complete_path = dir + "current_epoch_{}.pt".format(epoch)
    elif is_best:
        complete_path = dir + "model_best.pt"
    else:
        complete_path = dir + "current_epoch.pt"

    # if is_best:
    #     for file in os.listdir(dir):
    #         if fnmatch.fnmatch(file, "model_best*"):
    #             os.remove(dir + "/" + file)
    #             break
    # else:
    #     if not keep_checkpoints:
    #         for file in os.listdir(dir):
    #             if fnmatch.fnmatch(file, "current_*"):
    #                 os.remove(dir + "/" + file)
    #                 break
    print("Saving {}".format(complete_path))
    torch.save(state, complete_path)
    wandb.save(complete_path)

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
    with open(dir + "/params.json", "w") as outfile:
        json.dump(args, outfile, indent=4)
    wandb.save(dir + "/params.json")

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

def create_experiment_folder(config, run_name):
    return os.path.join(config.experiment_name, str(run_name))

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def update_output_runs(output, dataset, run_id, seed, experiment_name, checkpoint=None):
    output_path = pathlib.Path(output).parent
    pruning = experiment_name.split("/")[-1]
    if pruning == "pruning":
        if checkpoint is None:
            checkpoint = "current_epoch.pt"
    else:
        checkpoint = "model_best.pt"
    if os.path.isfile(output):
        f = open(output, 'r')
        dictionary = json.load(f)
        f.close()
    else :
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        dictionary = {}
        dictionary["experiment_name"] = experiment_name.split("/")[0]
        dictionary["datasets"] = {}

    if dataset not in dictionary["datasets"]:
        dictionary["datasets"][dataset] = {}
    if str(seed) not in dictionary["datasets"][dataset]:
        dictionary["datasets"][dataset][str(seed)] = {}
    dictionary["datasets"][dataset][str(seed)][pruning] = {"run" : run_id, "checkpoint": checkpoint, "config": "params.json"}

    print(dictionary)
    f = open(output, 'w+')
    f.write(json.dumps(dictionary))
    f.close()

# https://github.com/owenlo/RLE-Python/blob/master/rle.py
def encode(sequence):
    """Encode a sequence of characters and return the result as a list of tuples (data value and number of observed instances of value).
    Keyword arguments:
    sequence -- a sequence of characters to encode represented as a string.
    """
    count = 1
    result = []

    for x,item in enumerate(sequence): 
        if x == 0:
            continue
        elif item == sequence[x - 1]:
            count += 1
        else:        
            result.append((sequence[x - 1], count))
            count = 1            
    
    result.append((sequence[len(sequence) - 1], count))

    return result

def decode(sequence):
    """Decodes the sequence and returns the result as a string.
    Keyword arguments:
    sequence -- a list of tuples (data value and number of observed instances of value).
    """
    result = []

    for item in sequence:
        result.append(item[0] * item[1])

    return "".join(result)

def formatOutput(sequence):
    """Returns a print friendly version of the encoded data. 
    Keyword arguments:
    sequence -- list of tuples (data value and number of observed instances of value).
    """
    result = []

    for item in sequence:
        if (item[1] == 1):
            result.append(item[0])
        else:
            result.append(str(item[1]) + item[0])

    return "".join(result)


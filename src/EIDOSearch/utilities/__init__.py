#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def save_and_zip_model(model: torch.nn.Module, path: str) -> None:
    """Saves a `torch.nn.Module` instances' state_dict to file and tries to zip it. If successful it deletes the non-zipped file.

    Args:
        model (torch.nn.Module): `torch.nn.Module` instance.
        path (str): string that defines the path where to save the model.

    Returns:
        None:
    """
    torch.save(model.state_dict(), path)
    
    try:
        zip_path = path.split(os.sep)
        zip_path[-1] = zip_path[-1].replace("pt", "zip")
        zip_path = os.path.join(*zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_LZMA) as zip_file:
            zip_file.write(path, path.split(os.sep)[-1])
        os.remove(path)
    except Exception as ex:
        print("Error zipping model {}\n"
              "{}".format(path, ex))


def select_device(device: str) -> torch.device:
    """Selects and returns a `torch.device` device of type either cuda or cpu.

    Args:
        device (str): string containing the required device. Can be either 'cpu' for a cpu device or '0' or '0,1,2,3,etc.'
        for one or multiple cuda device(s).

    Returns:
        torch.device: selected device.
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
    
    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:0' if cuda else 'cpu')

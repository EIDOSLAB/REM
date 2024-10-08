import imp
from operator import imod
import torch
import numpy as np
from dataloaders.multiMNIST import MultiMNIST
import ops.utils as utils
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from dataloaders.affNIST import AffNISTDataset
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

def get_dataloader(config, directory):
    """Return train and test loaders for config.dataset."""
    if config.dataset == "mnist":
        return get_dataloaders_mnist(config, directory)
    elif config.dataset == "fashion-mnist":
        return get_dataloaders_fashion_mnist(config, directory)
    elif config.dataset == "cifar10":
        return get_dataloaders_cifar10(config, directory)
    elif config.dataset == "affNIST":
        return get_dataloaders_affNIST(config, directory)
    elif config.dataset == "tiny-imagenet-200":
        return get_dataloaders_tiny_imagenet_200(config, directory)
    elif config.dataset == "multiMNIST":
        return get_dataloaders_multiMNIST(config, directory)


def get_dataloaders_mnist(config, directory):
    """Return train, val and test loaders for MNIST."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=(config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(size=(config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.MNIST(root=directory + "/MNIST/raw/",
                                   train=True,
                                   transform=train_transform,
                                   download=True)

    valid_dataset = datasets.MNIST(root=directory + "/MNIST/raw/",
                                   train=True,
                                   transform=test_transform,
                                   download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.MNIST(root=directory + "/MNIST/raw/",
                                  train=False,
                                  transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader

def get_dataloaders_fashion_mnist(config, directory):
    """Return train, val and test loaders for Fashion-MNIST."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=(config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(size=(config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.FashionMNIST(root=directory + "/FashionMNIST/raw/",
                                          train=True,
                                          transform=train_transform,
                                          download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_dataset = datasets.FashionMNIST(root=directory + "/FashionMNIST/raw/",
                                          train=True,
                                          transform=test_transform,
                                          download=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.FashionMNIST(root=directory + "/FashionMNIST/raw/",
                                         train=False,
                                         transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10(config, directory):
    """Return train, val and test loaders for CIFAR10."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((config.input_height, config.input_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            #https://github.com/Sarasra/models/blob/master/research/capsules/input_data/cifar10/cifar10_input.py#L80
            transforms.ColorJitter(brightness=0.25, contrast=(0.2, 1.8), saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
    ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    test_transform = transforms.Compose([
        transforms.Resize((config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.CIFAR10(root=directory + "/CIFAR10/raw/",
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    indices = list(range(len(train_dataset)))
    split = 5500
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_dataset = datasets.CIFAR10(root=directory + "/CIFAR10/raw/",
                                     train=True,
                                     transform=test_transform,
                                     download=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.CIFAR10(root=directory + "/CIFAR10/raw/",
                                    train=False,
                                    transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))
    return train_loader, valid_loader, test_loader


def get_dataloaders_affNIST(config, directory):
    """Return train, val and test loaders for affNIST."""

    # Transforms
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                size=(config.input_height, config.input_width), padding=config.shift_pixels),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Pad(int(config.shift_pixels/2)),
            transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Pad(int(config.shift_pixels/2)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Datasets
    train_dataset = datasets.MNIST(root=directory + "/MNIST/raw/",
                                   train=True,
                                   transform=train_transform,
                                   download=True)

    val_dataset = datasets.MNIST(root=directory + "/MNIST/raw/",
                                 train=False,
                                 transform=val_transform)

    test_dataset = AffNISTDataset(
        "../data/raw/affNIST/test.mat", test_transform)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=config.pin_memory,
                                             worker_init_fn=lambda id: utils.set_seed(42))

    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, pin_memory=config.pin_memory, worker_init_fn=lambda id: utils.set_seed(42))

    return train_loader, val_loader, test_loader


def get_dataloaders_tiny_imagenet_200(config, directory):
    """Return train, val and test loaders for Tiny ImageNet."""
    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((config.input_height, config.input_width)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
    ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config.input_height, config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

    test_transform = transforms.Compose([
        transforms.Resize((config.input_height, config.input_width)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    train_dataset = datasets.ImageFolder(directory + "/tiny-imagenet-200/train", train_transform)

    indices = list(range(len(train_dataset)))
    split = 10000
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    valid_dataset = datasets.ImageFolder(directory + "/tiny-imagenet-200/train", test_transform)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))
    test_dataset = datasets.ImageFolder(directory + "/tiny-imagenet-200/test", test_transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             worker_init_fn=lambda id: utils.set_seed(42))

    return train_loader, valid_loader, test_loader


def get_dataloaders_multiMNIST(config, directory):
    """Return train, val and test loaders for affNIST."""

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    # Datasets
    train_dataset = MultiMNIST(root=directory + '/MultiMNIST/raw', train=True, download=True, transform=train_transform)
    test_dataset = MultiMNIST(root=directory + '/MultiMNIST/raw', train=False, download=True, transform=test_transform)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers,
                                               pin_memory=config.pin_memory,
                                               worker_init_fn=lambda id: utils.set_seed(42))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=config.pin_memory,
                                             worker_init_fn=lambda id: utils.set_seed(42))

    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, pin_memory=config.pin_memory, worker_init_fn=lambda id: utils.set_seed(42))

    return train_loader, test_loader, test_loader


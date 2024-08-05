import os
import torch
import logging
import json
import wandb
import argparse
import loss.capsule_loss as cl
from models.resNetCapsNet import ResNet18VectorCapsNet
import ops.utils as utils
import torch.nn as nn
from os.path import dirname, abspath
from ops.utils import save_args
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter
from EIDOSearch.evaluation import architecture_stat
from EIDOSearch.regularizers import LOBSTER
from EIDOSearch.pruning import PlateauIdentifier, find_best_unstructured_magnitude_threshold
from layers.capsule import LinearCaps2d

wandb.login()
# Opening wandb file
f = open('wandb_project.json',)
wand_settings = json.load(f)

def train_test_caps(config, args):
    run = wandb.init(project=wand_settings["project"], entity=wand_settings["entity"], reinit=True)
    print(wandb.run.name)
    wandb.config.update(config)
    checkpoint_path = args.checkpoint
    experiment_folder = utils.create_experiment_folder(config, wandb.run.name)

    utils.set_seed(config.seed)
    base_dir = dirname(dirname(abspath(__file__)))

    test_base_dir = base_dir + "/results/" + config.dataset + "/" + config.model + "/" + experiment_folder

    logdir = test_base_dir + "/logs/"
    checkpointsdir = test_base_dir + "/checkpoints/"
    runsdir = test_base_dir + "/runs/"
    imgdir = test_base_dir + "/images/"

    # Make model checkpoint directory
    if not os.path.exists(checkpointsdir):
        os.makedirs(checkpointsdir)

    # Make log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Make img directory
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    # Set logger path
    utils.set_logger(os.path.join(logdir, "model.log"))

    # Get dataset loaders
    train_loader, valid_loader, test_loader = get_dataloader(config, base_dir)

    # Enable GPU usage
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device(config.cuda_device)
    else:
        device = torch.device("cpu")

    caps_model = VectorCapsNet(config, device)
    #caps_model = ResNet18VectorCapsNet(config, device)
    wandb.watch(caps_model, log="all")

    utils.summary(caps_model, config)

    caps_criterion = cl.CapsLoss(config.caps_loss,
                                 config.margin_loss_lambda,
                                 config.reconstruction_loss_lambda,
                                 config.batch_averaged,
                                 config.reconstruction is not None,
                                 config.m_plus,
                                 config.m_minus,
                                 config.m_min,
                                 config.m_max,
                                 device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    caps_model.to(device)
    caps_model.load_state_dict(checkpoint["state_dict"], strict=False)

    for param in caps_model.parameters():
        param.requires_grad = False

    for param in caps_model.decoder.parameters():
        param.requires_grad = True

    if config.optimizer == "adam":
        caps_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    else:
        caps_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    caps_scheduler = torch.optim.lr_scheduler.ExponentialLR(caps_optimizer, config.decay_rate)

    # LOBSTER
    pruning_layers = (nn.Conv2d, LinearCaps2d, nn.Linear)
    LOBSTER_optimizer = LOBSTER(caps_model, config.pruning["args"]["max_lmbda"], pruning_layers)
    #LOBSTER_optimizer = None
    plateau_identifier = PlateauIdentifier(caps_model, config.pruning["args"]["pwe"])

    caps_model.to(device)
    print("MODULES")
    for n, m in caps_model.named_modules():
        print(n, m.__class__)

    for state in caps_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Print the model architecture and parameters
    utils.summary(caps_model, config)

    # Save current settings (hyperparameters etc.)
    save_args(config, test_base_dir)

    # Writer for TensorBoard
    writer = None
    if config.tensorboard:
        writer = SummaryWriter(runsdir)

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    logging.info("Initial learning rate: {:.4f}".format(caps_scheduler.get_last_lr()[0]))
    logging.info("Number of routing iterations: {}".format(config.num_routing_iterations))

    best_loss = float('inf')

    epoch = 0
    best_epoch = 0
    training = True
    while training:
        # Start training
        logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
        train(logging, config, train_loader, caps_model, caps_criterion, caps_optimizer, caps_scheduler, LOBSTER_optimizer, pruning_layers, writer, epoch, device)

        # Start validation
        val_loss, val_acc = test(logging, config, valid_loader, caps_model, caps_criterion, writer, epoch, device,
                         imgdir, split="validation")
        print(val_acc)
        # Start testing
        test_loss, test_acc = test(logging, config, test_loader, caps_model, caps_criterion, writer, epoch, device, imgdir, split="test")

        if config.dataset != "affNIST" and plateau_identifier(val_loss, epoch):
            T = find_best_unstructured_magnitude_threshold(caps_model, valid_loader, caps_criterion, 0, "min",
                                                           "global", pruning_layers, device, 1,
                                                           False, True, ["weight", "logits"])        
        if config.dataset == "affNIST": 
            # PRUNING
            if config.pruning["args"]["max_lmbda"] > 0 and val_acc > 0.5 and plateau_identifier(val_loss, epoch):
                T = find_best_unstructured_magnitude_threshold(caps_model, valid_loader, caps_criterion, 0, "min",
                                                            "global", pruning_layers, device, 1,
                                                            False, True, ["weight", "logits"])
            # NO PRUNING
            elif config.pruning["args"]["max_lmbda"] == 0 and plateau_identifier(val_loss, epoch):
                T = find_best_unstructured_magnitude_threshold(caps_model, valid_loader, caps_criterion, 0, "min",
                                                            "global", pruning_layers, device, 1,
                                                            False, True, ["weight", "logits"])

        
        if writer:
            writer.add_scalar('routing/iterations', caps_model.classCaps.num_iterations, epoch)
            if config.caps_loss == "spread":
                writer.add_scalar('spread_loss/margin', caps_criterion.caps_loss.margin, epoch)
                wandb.log({'spread_loss/margin': caps_criterion.caps_loss.margin}, step=epoch)
            writer.add_scalar('lr', caps_scheduler.get_last_lr()[0], epoch)

        wandb.log({'routing/iterations': caps_model.classCaps.num_iterations}, step=epoch)

        formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))
        checkpoint_filename = "epoch_{}-it{}-val_loss_{:.6f}-val_acc_{:.6f}-test_loss_{:.6f}-test_acc_{:.6f}-lr{}_".format(
                                                                            formatted_epoch, 
                                                                            caps_model.classCaps.num_iterations,
                                                                            val_loss, 
                                                                            val_acc,
                                                                            test_loss, 
                                                                            test_acc,
                                                                            caps_scheduler.get_last_lr()[0])

        if val_loss < best_loss:
            utils.save_checkpoint({
                "epoch": epoch,
                "routing_iterations": caps_model.classCaps.num_iterations,
                "state_dict": caps_model.state_dict(),
                "metric": config.monitor,
                "optimizer": caps_optimizer.state_dict(),
                "scheduler": caps_scheduler.state_dict(),
            }, True, checkpointsdir, checkpoint_filename)
            best_epoch = epoch
            best_loss = val_loss
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_test_acc"] = test_acc
            wandb.run.summary["best_test_loss"] = test_loss
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_val_loss"] = val_loss

        # Save current epoch checkpoint
        utils.save_checkpoint({
            "epoch": epoch,
            "routing_iterations": caps_model.classCaps.num_iterations,
            "state_dict": caps_model.state_dict(),
            "metric": config.monitor,
            "optimizer": caps_optimizer.state_dict(),
            "scheduler": caps_scheduler.state_dict(),
        }, False, checkpointsdir, checkpoint_filename, config.dataset=="mnist" and config.reconstruction=="None" and config.seed==42)
        epoch += 1

        # Get pruning statistics
        stat = architecture_stat(caps_model, pruning_layers)
        for k, v in stat["layer_param_non_zero_perc"].items():
                wandb.log({"pruning/layer_param_non_zero_perc_"+k: v}, step=epoch)
        for k, v in stat["layer_param_non_zero"].items():
                wandb.log({"pruning/layer_param_non_zero_"+k: v}, step=epoch)
        wandb.log({"pruning/network_param_non_zero_perc": stat["network_param_non_zero_perc"]}, step=epoch)
        wandb.log({"pruning/network_param_non_zero": stat["network_param_non_zero"]}, step=epoch)
        # if config.dataset == "affNIST":
        #     if abs(val_acc - 0.9923) <= 0.0001:
        #         training = False
        # elif epoch - best_epoch > config.patience:
        #     training = False
        if epoch - best_epoch > config.patience:
            training = False
    if writer:
        writer.close()
    wandb.save(checkpointsdir+"/*.pt")
    run.finish()

def main(config, args):
    for k in range(len(config.seeds)):
        config.seed = config.seeds[k]
        train_test_caps(config, args)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint.pt path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = utils.DotDict(json.load(open(args.config)))
    main(config, args)
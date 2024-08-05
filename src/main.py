import os
import torch
import time
import logging
import json
import wandb
import argparse
import loss.capsule_loss as cl
import ops.utils as utils
import torch.nn as nn
from models.mobilev1CapsNet import Mobilev1CapsNet
from models.resNet50CapsNet import ResNet50VectorCapsNet
from models.vectorConvCapsNet import vectorConvCapsNet
from models.gammaCapsNet import GammaCapsNet
from models.deepCapsNet import DeepCapsNet, FC_Caps
from ops.utils import save_args
from ops.utils import update_output_runs
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from train import train
from test import test
from models.resNetCapsNet import ResNet18VectorCapsNet
from torch.utils.tensorboard import SummaryWriter
from EIDOSearch.evaluation import architecture_stat
from EIDOSearch.regularizers import LOBSTER
from EIDOSearch.pruning import PlateauIdentifier, find_best_unstructured_magnitude_threshold
from layers.capsule import LinearCaps2d
wandb.login()

def train_test_caps(config, output):
    # Opening wandb file
    f = open('wandb_project.json')
    wand_settings = json.load(f)
    run = wandb.init(project=wand_settings["project"], entity=wand_settings["entity"], reinit=True)
    wandb.config.update(config)

    update_output_runs(output, config.dataset, wandb.run.id, config.seed, config.experiment_name)

    experiment_folder = utils.create_experiment_folder(config, wandb.run.id)

    utils.set_seed(config.seed)
    base_dir = "/data"

    test_base_dir = "results/" + config.dataset + "/" + experiment_folder

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

    if config.model == "ResNetVectorCapsNet":
        caps_model = ResNet18VectorCapsNet(config, device)
    elif config.model == "ResNet50VectorCapsNet":
        caps_model = ResNet50VectorCapsNet(config, device)
    elif config.model == "Mobilev1CapsNet":
        caps_model = Mobilev1CapsNet(config, device)
    elif config.model == "vectorConvCapsNet":
        caps_model = vectorConvCapsNet(config, device)
    elif config.model == "GammaCapsNet":
        caps_model = GammaCapsNet(config, device)
    elif config.model == "DeepCapsNet":
        caps_model = DeepCapsNet(config, device)
    else:
        caps_model = VectorCapsNet(config, device)
    wandb.watch(caps_model, log="all")

    utils.summary(caps_model, config, config.model)

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

    if config.optimizer == "adam":
        caps_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    else:
        caps_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    caps_scheduler = torch.optim.lr_scheduler.ExponentialLR(caps_optimizer, config.decay_rate)

    # LOBSTER
    pruning_layers = (nn.Conv2d, nn.Conv3d, LinearCaps2d, FC_Caps)
    LOBSTER_optimizer = LOBSTER(caps_model, config.pruning["args"]["max_lmbda"], pruning_layers)
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
    utils.summary(caps_model, config, config.model)

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
        
        start_training_time = time.time() 
        train(logging, config, train_loader, caps_model, caps_criterion, caps_optimizer, caps_scheduler, LOBSTER_optimizer, pruning_layers, writer, epoch, device)
        end_training_time = time.time()
        tot_training_time = end_training_time - start_training_time
        wandb.log({'training_time': tot_training_time}, step=epoch)
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
            writer.add_scalar('lr', caps_scheduler.get_last_lr()[0], epoch)

        wandb.log({'routing/iterations': caps_model.classCaps.num_iterations}, step=epoch)

        formatted_epoch = str(epoch).zfill(len(str(config.epochs))+1)
        update_output_runs(output, config.dataset, wandb.run.id, config.seed, config.experiment_name, "current_epoch_{}.pt".format(formatted_epoch))
        
        checkpoint_filename = "epoch_{}".format(formatted_epoch)

        if val_loss < best_loss:
            utils.save_checkpoint({
                "epoch": epoch,
                "routing_iterations": caps_model.classCaps.num_iterations,
                "state_dict": caps_model.state_dict(),
                "metric": config.monitor
                #"optimizer": caps_optimizer.state_dict(),
                #"scheduler": caps_scheduler.state_dict(),
            }, True, checkpointsdir, checkpoint_filename, formatted_epoch)
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
            "metric": config.monitor
            #"optimizer": caps_optimizer.state_dict(),
            #"scheduler": caps_scheduler.state_dict(),
        }, False, checkpointsdir, checkpoint_filename, formatted_epoch, config.dataset=="mnist" and config.reconstruction=="None" and config.seed==42)
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

        if epoch == 100 and config.model == "DeepCapsNet":
            #print("Hard margin loss")
            caps_criterion.caps_loss.margin_loss_lambda = 0.8
            caps_criterion.caps_loss.m_plus = 0.95
            caps_criterion.caps_loss.m_minus = 0.05
    if writer:
        writer.close()
    wandb.save(checkpointsdir+"/*.pt")
    run.finish()

def main(config, output):
    for k in range(len(config.seeds)):
        config.seed = config.seeds[k]
        train_test_caps(config, output)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--output', default=None, type=str, help='wandb.json file to track runs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = utils.DotDict(json.load(open(args.config)))
    main(config, args.output)
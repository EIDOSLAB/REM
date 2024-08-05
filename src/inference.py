import os
import pickle
import torch
import time
import logging
import json
import wandb
import argparse
import loss.capsule_loss as cl
import ops.caps_utils as caps_utils
import ops.utils as utils
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn
from tqdm import *
from scipy.stats import entropy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from models.resNetCapsNet import ResNet18VectorCapsNet
from models.mobilev1CapsNet import Mobilev1CapsNet
from models.resNet50CapsNet import ResNet50VectorCapsNet
from models.vectorConvCapsNet import vectorConvCapsNet
from models.gammaCapsNet import GammaCapsNet
from models.deepCapsNet import DeepCapsNet

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

wandb.login()
api = wandb.Api()
# Opening wandb file
f = open('wandb_project.json',)
wand_settings = json.load(f)

def main(config, bins, save_img=False):
    base_dir = "/data"

    # Set logger
    utils.set_logger(None)
    utils.set_seed(config.seed)
    # Get dataset loaders
    train_loader, valid_loader, test_loader = get_dataloader(config, base_dir)
    if config.split == 'test':
        loader = test_loader
    elif config.split == 'validation':
        loader = valid_loader
    else:
        loader = train_loader
    # Enable GPU usage
    device = torch.device("cuda:0")

    if config.model == "ResNetVectorCapsNet":
        caps_model = ResNet18VectorCapsNet(config, device, bins)
    elif config.model == "ResNet50VectorCapsNet":
        caps_model = ResNet50VectorCapsNet(config, device, bins)
    elif config.model == "Mobilev1CapsNet":
        caps_model = Mobilev1CapsNet(config, device, bins)
    elif config.model == "vectorConvCapsNet":
        caps_model = vectorConvCapsNet(config, device, bins)
    elif config.model == "GammaCapsNet":
        caps_model = GammaCapsNet(config, device, bins)
    elif config.model == "DeepCapsNet":
        caps_model = DeepCapsNet(config, device, bins)
    else:
        caps_model = VectorCapsNet(config, device)
    
    # Print the model architecture and parameters
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
    model_wandb = wandb.restore(str(config.checkpoint), run_path=os.path.join(wand_settings["entity"],wand_settings["project"], str(config.run_id)))
    checkpoint = torch.load(model_wandb.name, map_location='cpu')
    caps_model.to(device)
    caps_model.load_state_dict(checkpoint["state_dict"])

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
    print("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))

    # Start testing
    test_loss, test_acc, dict_cij = test(logging, config, loader, caps_model, caps_criterion, None, 0, device, None, bins, split="test", save_img=save_img)
    return test_acc, dict_cij

def test(logging, config, loader, model, criterion, writer, epoch, device, imgdir, bins, split, save_img):
    dict_cij = [{} for _ in range(config.num_classes)]
    loss = 0
    tot_samples = len(loader.sampler)
    
    if config.reconstruction is not None:
        margin_loss = 0
        recons_loss = 0
    precision = np.zeros(config.num_classes)
    recall = np.zeros(config.num_classes)
    f1score = np.zeros(config.num_classes)
    balanced_accuracy = 0

    correct = 0

    labels = range(config.num_classes)
    all_labels = []
    all_pred = []

    model.eval()

    start_time = time.time()
    batch_index = 0
    for data, target in tqdm(loader):
        # Store the indices for calculating accuracy
        label = target.unsqueeze(0).type(torch.LongTensor)

        batch_size = data.size(0)
        # Transform to one-hot indices: [batch_size, 10]
        target_encoded = F.one_hot(target, config.num_classes)

        # Use GPU if available
        data, target_encoded = data.to(device), target_encoded.to(device)

        # Output predictions
        if config.reconstruction != "None":
            if config.coupl_coeff:
                class_caps_poses, class_caps_activations, coupling_coefficients, reconstructions = model(data)
            else:
                class_caps_poses, class_caps_activations, reconstructions = model(data)
            c_loss, m_loss, r_loss = criterion(class_caps_activations, target_encoded, data, reconstructions)

            margin_loss += m_loss.item()
            recons_loss += r_loss.item()

        else:
            if config.coupl_coeff:
                class_caps_poses, class_caps_activations, coupling_coefficients = model(data)
            else:
                class_caps_poses, class_caps_activations = model(data)
            c_loss = criterion(class_caps_activations, target_encoded)

        loss += c_loss.item()
        pred = class_caps_activations.max(1, keepdim=True)[1].type(torch.LongTensor)
        top_activation = class_caps_activations.max(1)[0]

        if save_img:
            for i in range(config.num_classes): 
                caps_utils.save_dict_images(coupling_coefficients.detach(), data, pred, label, i, bins)

        # Compute dictionaries of cij for each class
        for i in range(config.num_classes): 
            caps_utils.compute_dict_coupl_coeff(dict_cij[i],coupling_coefficients.detach(), pred, label, i, bins)
 
        correct += pred.eq(label.view_as(pred)).cpu().sum().item()

        # Classification report
        label_flat = label.view(-1)
        pred_flat = pred.view(-1)
        all_labels.append(label_flat)
        all_pred.append(pred_flat)
        recall += recall_score(label_flat, pred_flat, labels=labels, average=None)
        precision += precision_score(label_flat, pred_flat, labels=labels, average=None)
        f1score += f1_score(label_flat, pred_flat, labels=labels, average=None)
        balanced_accuracy += balanced_accuracy_score(label_flat, pred_flat)

        batch_index += 1
    # Print time elapsed for every epoch
    end_time = time.time()
    formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))
    logging.info('\nEpoch {} takes {:.0f} seconds for {}.'.format(formatted_epoch, end_time - start_time, split))

    if config.reconstruction != "None":
        # Visualize reconstructed images of the last batch
        num_img = 8
        if num_img > batch_size:
            num_img = batch_size
        reconstructions = reconstructions[0:num_img].view(num_img, config.input_channels, config.input_height, config.input_width)
        originals = data[0:num_img].view(num_img, config.input_channels, config.input_height, config.input_width)
        images = torch.cat((originals, reconstructions), 0)
        images = vutils.make_grid(images, normalize=False, scale_each=False)
        #vutils.save_image(images, imgdir + '/img-epoch_{}.png'.format(formatted_epoch), normalize=False, scale_each=False)

    # Log test losses
    loss /= len(loader)
    if config.reconstruction != "None":
        margin_loss /= len(loader)
        recons_loss /= len(loader)
    acc = correct / tot_samples

    # Print test losses
    if config.reconstruction != "None":
        if config.caps_loss == "margin":
            logging.info("{} loss: {:.4f} Margin loss: {:.4f} Recons loss: {:.4f}".format(split, loss, margin_loss, recons_loss))
        else:
            logging.info("{} loss: {:.4f} Spread loss: {:.4f} Recons loss: {:.4f}".format(split, loss, margin_loss, recons_loss))

    else:
        logging.info("{} loss: {:.4f}".format(split, loss))

    # Log classification report
    recall /= len(loader)
    precision /= len(loader)
    f1score /= len(loader)
    balanced_accuracy /= len(loader)
    if writer:
        writer.add_scalar('{}/balanced_accuracy'.format(split), balanced_accuracy, epoch)

    # Print classification report
    logging.info("{} classification report:".format(split))
    for i in range(config.num_classes):
        logging.info("Class: {} Recall: {:.4f} Precision: {:.4f} F1-Score: {:.4f}".format(i,
                                                                                          recall[i],
                                                                                          precision[i],
                                                                                          f1score[i]))

    logging.info(confusion_matrix(torch.cat(all_labels), torch.cat(all_pred)))

    logging.info("{} accuracy: {}/{} ({:.2f}%)".format(split, correct, tot_samples,
                                                    100. * correct / tot_samples))
    logging.info("{} error: {}/{} ({:.2f}%)\n".format(split, tot_samples - correct,  tot_samples,
                                                 100. * (1 - correct / tot_samples)))

    return loss, acc, dict_cij

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--wandb_runs', default="wandb_runs.json", type=str, help='runs path')
    parser.add_argument('--output_dictionary', default=None, type=str, help='dictionary output path')
    parser.add_argument('--dataset', default="mnist", type=str, help='Dataset')
    parser.add_argument('--split', default="training", type=str, help='Dataset split')
    parser.add_argument('--binning', dest='binning', action='store_true')
    parser.add_argument('--no-binning', dest='binning', action='store_false')
    parser.set_defaults(binning=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    dataset = args.dataset
    binning = args.binning
    runs_path = args.wandb_runs
    split = args.split
    output_dictionary = args.output_dictionary

    with open(runs_path) as f:
        checkpoints_dict = json.load(f)


    Path(output_dictionary).mkdir(parents=True, exist_ok=True)
    #Path("dump/{}_rebuttal2/no-pruning/".format(args.dataset)).mkdir(parents=True, exist_ok=True)
    #Path("dump/{}_rebuttal2/pruning/".format(args.dataset)).mkdir(parents=True, exist_ok=True)

    accuracies_pruning = []
    accuracies_nopruning = []
    dict_cij_nopruning_count = []
    dict_cij_pruning_count = []
    network_param_zero_perc = []

    num_seeds = len(checkpoints_dict["datasets"][dataset].keys())
    print(num_seeds)
    if dataset == "tiny-imagenet-200":
        classes = 200
        bins = np.linspace(0., 1., num=129, endpoint=True)
    else:
        classes = 10
        bins = np.linspace(-0.05, 1.05, num=12, endpoint=True)
        #bins =  np.linspace(-0.10, 1.10, num=7, endpoint=True)

    #bins = np.linspace(-0.125, 1.125, num=6, endpoint=True) #centers [0., 0.25, 0.5 , 0.75, 1.  ]
    #bins = np.linspace(-0.25, 1.25, num=4, endpoint=True) #centers [0. , 0.5, 1. ]
    entropies_pruning = np.zeros(classes)
    entropies_nopruning = np.zeros(classes)

    for key in checkpoints_dict["datasets"][dataset].keys():
        run_id_nopruning = checkpoints_dict["datasets"][dataset][key]["no-pruning"]["run"]
        run_id_pruning = checkpoints_dict["datasets"][dataset][key]["pruning"]["run"]

        run_nopruning = api.run("{}/{}/{}".format(wand_settings["entity"],wand_settings["project"],run_id_nopruning))
        run_pruning = api.run("{}/{}/{}".format(wand_settings["entity"],wand_settings["project"],run_id_pruning))
        network_param_zero_perc.append(100-run_pruning.summary["pruning/network_param_non_zero_perc"])
        run_name_nopruning = run_nopruning.id
        run_name_pruning = run_pruning.id
        checkpoint_pruning = os.path.join("results/{}/".format(dataset), checkpoints_dict["experiment_name"], "pruning")
        checkpoint_no_pruning = os.path.join("results/{}/".format(dataset), checkpoints_dict["experiment_name"], "no-pruning")
        checkpoint_nopruning_path = os.path.join(checkpoint_no_pruning, run_name_nopruning, "checkpoints", checkpoints_dict["datasets"][dataset][key]["no-pruning"]["checkpoint"])
        f = wandb.restore(os.path.join(checkpoint_no_pruning, run_name_nopruning, checkpoints_dict["datasets"][dataset][key]["no-pruning"]["config"]), run_path=os.path.join(wand_settings["entity"],wand_settings["project"],run_id_nopruning)) 
        config_nopruning = utils.DotDict(json.load(open(f.name)))
        config_nopruning.checkpoint = checkpoint_nopruning_path
        config_nopruning.run_id = run_id_nopruning
        config_nopruning.binning = binning
        config_nopruning.split = split
        config_nopruning.augmentation = False
        config_nopruning.batch_size = 128
        checkpoint_pruning_path = os.path.join(checkpoint_pruning, run_name_pruning, "checkpoints", checkpoints_dict["datasets"][dataset][key]["pruning"]["checkpoint"])
        #checkpoint_pruning_path = os.path.join(checkpoint_pruning, run_name_pruning, "checkpoints", "current_epoch.pt")
        
        f = wandb.restore(os.path.join(checkpoint_pruning, run_name_pruning, checkpoints_dict["datasets"][dataset][key]["pruning"]["config"]), run_path=os.path.join(wand_settings["entity"],wand_settings["project"],run_id_pruning))
        config_pruning = utils.DotDict(json.load(open(f.name)))
        config_pruning.checkpoint = checkpoint_pruning_path
        config_pruning.run_id = run_id_pruning
        config_pruning.binning = binning
        config_pruning.split = split
        config_pruning.augmentation = False
        config_pruning.batch_size = 128
        acc_pruning = []
        acc_nopruning = []

        #save_images = dataset == "mnist" and key=="42"
        save_images = False

        # NO PRUNING
        acc_nopruning, dict_cij_nopruning = main(config_nopruning, bins, False)
        accuracies_nopruning.append(acc_nopruning)
        # PRUNING
        acc_pruning, dict_cij_pruning = main(config_pruning, bins, save_images)
        accuracies_pruning.append(acc_pruning)

        if output_dictionary is not None:
            a_file = open("{}/{}_dict_cij_nopruning_nodecoder_{}.pkl".format(output_dictionary, dataset, split), "wb")
            pickle.dump(dict_cij_nopruning, a_file)
            a_file.close()
            a_file = open("{}/{}_dict_cij_pruning_nodecoder_{}.pkl".format(output_dictionary, dataset, split), "wb")
            pickle.dump(dict_cij_pruning, a_file)
            a_file.close()

        dict_pruning = []
        dict_nopruning = []
        for class_id in range(config_nopruning.num_classes): 
            dict_nopruning.append(len(dict_cij_nopruning[class_id].keys()))
            dict_pruning.append(len(dict_cij_pruning[class_id].keys()))
            counts_nopruning = [dict_cij_nopruning[class_id][key] for key in dict_cij_nopruning[class_id].keys()]
            counts_pruning = [dict_cij_pruning[class_id][key] for key in dict_cij_pruning[class_id].keys()]
            entropies_nopruning[class_id] += entropy(counts_nopruning, base=2)
            entropies_pruning[class_id] += entropy(counts_pruning, base=2)

        dict_cij_nopruning_count.append(dict_nopruning)
        dict_cij_pruning_count.append(dict_pruning)

entropies_nopruning = entropies_nopruning / num_seeds
entropies_pruning = entropies_pruning / num_seeds

print("(NP) Accuracy mean: {}".format(np.mean(accuracies_nopruning)))
print("(NP) Accuracy std: {}".format(np.std(accuracies_nopruning)))
print("(P) Accuracy mean: {}".format(np.mean(accuracies_pruning)))
print("(P) Accuracy std: {}".format(np.std(accuracies_pruning)))
print("(P) Median perc pruned parameters: {}".format(np.median(network_param_zero_perc)))


print("(NP) Entropies mean: {}".format(np.mean(entropies_nopruning)))
print("(NP) Entropies std: {}".format(np.std(entropies_nopruning)))
print("(P) Entropies mean: {}".format(np.mean(entropies_pruning)))
print("(P) Entropies std: {}".format(np.std(entropies_pruning)))

string_latex_pruning = ""
string_latex_nopruning = ""
for class_id in range(config_nopruning.num_classes): 
    print("Class ID {} (NP) Median number of keys in dictionary: {}".format(class_id, np.median(dict_cij_nopruning_count, axis=0)[class_id]))  
    print("Class ID {} (P) Median number of keys in dictionary: {}".format(class_id, np.median(dict_cij_pruning_count, axis=0)[class_id]))

    string_latex_nopruning = string_latex_nopruning + " & $" + str(int(np.median(dict_cij_nopruning_count, axis=0)[class_id])) + "$"
    string_latex_pruning = string_latex_pruning + " & $" + str(int(np.median(dict_cij_pruning_count, axis=0)[class_id])) + "$"

print(string_latex_nopruning)
print(string_latex_pruning)
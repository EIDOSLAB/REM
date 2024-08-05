import os
import torch
import logging
import json
import time
import wandb
import argparse
import warnings
import loss.capsule_loss as cl
import ops.utils as utils
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import entropy
from os.path import dirname, abspath
from ops.caps_utils import compute_dict_coupl_coeff, compute_entropy_caps, compute_entropy_coupl_coeff, estimation_coupl_coeff, get_prob_cij_k
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from models.resNetCapsNet import ResNet18VectorCapsNet
from tqdm import *
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from glob import glob

warnings.filterwarnings("ignore")

wandb.login()
api = wandb.Api()
# Opening wandb file
f = open('wandb_project.json',)
wand_settings = json.load(f)

def main(config, bins):
    base_dir = dirname(dirname(abspath(__file__)))

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

    caps_model = VectorCapsNet(config, device, bins)

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

    checkpoint = torch.load(str(config.checkpoint), map_location='cpu')
    caps_model.to(device)
    caps_model.load_state_dict(checkpoint["state_dict"])

    # Print the model architecture and parameters
    utils.summary(caps_model, config)

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
    print("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))

    # Start testing
    test_loss, test_acc, cij, estimated_cij, entropy_cij, poses, dict_cij = test(logging, config, loader, caps_model, caps_criterion, None, 0, device, None, bins, split="test")
    return test_acc, cij, estimated_cij, entropy_cij, poses, dict_cij

def test(logging, config, loader, model, criterion, writer, epoch, device, imgdir, bins, split):
    dict_cij = [{} for _ in range(config.num_classes)]
    loss = 0
    tot_samples = len(loader.sampler)
    cij = torch.zeros(config.num_classes, config.num_classes, model.num_primary_units).to(device)
    centers = (bins[1:] + bins[:-1])/2
    estimated_cij = torch.zeros(config.num_classes,1,1,config.num_primaryCaps_types,model.h1,model.w1).to(device)
    prob_cij_k = torch.zeros((len(centers), config.num_classes, model.num_primary_units)).to(device)
    poses = {}
    num_batches = len(loader)
    
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
        batch_correct_cij = compute_entropy_coupl_coeff(coupling_coefficients.detach(), pred, label)
        estimated_cij += estimation_coupl_coeff(coupling_coefficients.detach(), pred, label)
        prob_cij_k += get_prob_cij_k(tot_samples, centers, coupling_coefficients, pred, label)

        # Compute dictionaries of cij for each class
        for i in range(config.num_classes): 
            compute_dict_coupl_coeff(dict_cij[i],coupling_coefficients.detach(), pred, label, i, bins)
 
        cij += batch_correct_cij
        entropy_caps = compute_entropy_caps(class_caps_poses.detach(), pred, label)
        for i in range(config.num_classes):
            if i in poses:
                poses[i] = torch.cat((poses[i], entropy_caps[i]), dim=0)
            else:
                poses[i] = entropy_caps[i]
        for i in range(config.num_classes):
            poses[i] = poses[i].cpu().detach()
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

    cij = cij/num_batches
    estimated_cij = estimated_cij/num_batches

    prob_cij_k += 1e-10
    entropy_cij = - torch.sum(prob_cij_k*torch.log2(prob_cij_k), dim=0) #10x1152

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

    if config.coupl_coeff:
        return loss, acc, cij, estimated_cij, entropy_cij, poses, dict_cij
    else:
        return loss, acc

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--wandb_runs', default="wandb_runs.json", type=str, help='runs path')
    parser.add_argument('--dataset', default="mnist", type=str, help='Dataset')
    parser.add_argument('--output', default="../dump", type=str, help='Output path')
    parser.add_argument('--split', default="training", type=str, help='Dataset split')
    parser.add_argument('--epoch', default='best', type=str, help='First or last epoch')
    parser.add_argument('--pruning', dest='pruning', action='store_true')
    parser.add_argument('--no-pruning', dest='pruning', action='store_false')
    parser.set_defaults(pruning=True)
    args = parser.parse_args()
    return args
    
def find_checkpoint(folder_path, model):
    files = glob(os.path.join(folder_path, model))
    files.sort(key=os.path.getmtime)
    return files

if __name__ == "__main__":
    args = parse_arguments()
        
    dataset = args.dataset
    output = args.output
    binning = True
    runs_path = args.wandb_runs
    split = args.split
    dict_cij_count = []

    if args.pruning:
        pruning = "pruning"
        output_file = "{}/cij_pruning_epoch_{}.npy".format(output, args.epoch)
    else:
        pruning = "no-pruning"
        output_file = "{}/cij_nopruning_epoch_{}.npy".format(output, args.epoch)

    with open(runs_path) as f:
        checkpoints_dict = json.load(f)

    model_name = "VectorCapsNet"
    classes = 10
    bins = np.linspace(-0.05, 1.05, num=12, endpoint=True)

    key = list(checkpoints_dict["datasets"][dataset].keys())[0]
    run = api.run("{}/{}/{}".format(wand_settings["entity"], wand_settings["project"], checkpoints_dict["datasets"][dataset][key][pruning]["run"]))
    run_name  = run.name
    checkpoint = os.path.join("../results/{}/{}/".format(dataset,model_name), checkpoints_dict["experiment_name"], pruning)

    if args.epoch == "first":
        checkpoints_list = find_checkpoint(os.path.join(checkpoint, run_name, "checkpoints"), "current_epoch_00*")
    elif args.epoch == "last" or args.epoch == "best":
        checkpoints_list = find_checkpoint(os.path.join(checkpoint, run_name, "checkpoints"), checkpoints_dict["datasets"][dataset][key][pruning]["checkpoint"])
    checkpoints_list = [checkpoints_list[-1]]
    array_to_dump = []
    for index, checkpoint_path in enumerate(checkpoints_list):
        #if index % 10 == 0:
        entropies = np.zeros(classes)
        config_path = os.path.join(checkpoint, run_name, checkpoints_dict["datasets"][dataset][key][pruning]["config"])
        config = utils.DotDict(json.load(open(config_path)))
        config.checkpoint = checkpoint_path
        config.split = split
        config.augmentation = False
        config.batch_size = 512
        config.binning = binning

        # PRUNING
        acc, cij, _, _, _,dict_cij = main(config, bins)
        cij = cij.cpu().numpy()
        with open(output_file, 'wb') as f:
            np.save(f, cij)

        dict = []
        for class_id in range(config.num_classes): 
            dict.append(len(dict_cij[class_id].keys()))
            dict.append(len(dict_cij[class_id].keys()))
            counts_nopruning = [dict_cij[class_id][key] for key in dict_cij[class_id].keys()]
            counts_pruning = [dict_cij[class_id][key] for key in dict_cij[class_id].keys()]
            entropies[class_id] += entropy(counts_nopruning, base=2)

        dict_cij_count.append(dict)

    print("Accuracy: {}".format(acc))

    print("Entropy: {}".format(np.mean(entropies)))

    # string_latex = ""
    # for class_id in range(config.num_classes): 
    #     print("Class ID {} Median number of keys in dictionary: {}".format(class_id, np.median(dict_cij_count, axis=0)[class_id]))  

    #     string_latex = string_latex + " & $" + str(int(np.median(dict_cij_count, axis=0)[class_id])) + "$"
    #     print(entropies)

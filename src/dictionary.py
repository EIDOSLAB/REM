import pickle
import numpy as np
import pickle
import torch
import logging
import json
import argparse
import time
import matplotlib
import loss.capsule_loss as cl
import ops.utils as utils
import numpy as np
import torchvision.utils as vutils
import torch.nn.functional as F
from os.path import dirname, abspath
from ops.caps_utils import save_dict_images
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from tqdm import *
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
matplotlib.use('TkAgg')

def main(config, dict_cij, bins, output_folder):
    base_dir = dirname(dirname(abspath(__file__)))

    # Set logger
    utils.set_logger(None)
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

    caps_model = VectorCapsNet(config, device)

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

    caps_model.to(device)
    checkpoint = torch.load(str(config.checkpoint))
    caps_model.load_state_dict(checkpoint["state_dict"])

    # Print the model architecture and parameters
    utils.summary(caps_model, config)

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
    print("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))

    # Start testing
    test(dict_cij, logging, config, loader, caps_model, caps_criterion, None, 0, device, None, bins, output_folder, split="test")

def test(dict_cij, logging, config, loader, model, criterion, writer, epoch, device, imgdir, bins, output_folder, split):
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

        for i in range(config.num_classes): 
            save_dict_images(config, coupling_coefficients.detach(), data, pred, label, i, bins, config.dataset, output_folder)

        # Compute dictionaries of cij for each class
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

    return loss, acc

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--dataset', default="mnist", type=str, help='Dataset')
    parser.add_argument('--split', default="test", type=str, help='Dataset split')
    parser.add_argument('--dictionary', default="../dump/mnist_dict_cij_pruning_nodecoder_test.pkl", type=str, help='Dictionary pkl path')
    parser.add_argument('--output_folder', default="../dictionary", type=str, help='Checkpoint path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    bins =  np.linspace(-0.05, 1.05, num=12, endpoint=True)

    config = utils.DotDict(json.load(open(args.config)))
    config.checkpoint = args.checkpoint
    config.binning = True
    config.split = args.split

    a_file = open(args.dictionary, "rb")
    dict_cij = pickle.load(a_file)
    a_file.close()

    num_classes = config.num_classes
    for class_id in range(num_classes):
        cij = []
        print("Number of keys in dictionary {}: {}".format(class_id, len(dict_cij[class_id].keys())))
        for keys in dict_cij[class_id].keys():
            cij.append(dict_cij[class_id][keys])

        np_cij = np.array(cij)
        np_cij = np_cij/np.sum(np_cij) 

        entropy = -np.sum(np_cij*np.log2(np_cij))

        print(entropy)

    main(config, dict_cij, bins, args.output_folder)
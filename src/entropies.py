import os
import torch
import logging
import json
import argparse
import loss.capsule_loss as cl
from ops.caps_utils import compute_dict_coupl_coeff, compute_entropy_caps, compute_entropy_coupl_coeff, estimation_coupl_coeff, get_prob_cij_k
import ops.utils as utils
import numpy as np
from os.path import dirname, abspath
from dataloaders.load_data import get_dataloader
from models.vectorCapsNet import VectorCapsNet
from models.resNetCapsNet import ResNet18VectorCapsNet
from tqdm import *
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import wandb
from glob import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
sns.set_style("dark")


warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })

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

    if config.model == "ResNetVectorCapsNet":
        caps_model = ResNet18VectorCapsNet(config, device, bins)
    else:
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

    return loss, acc, cij, estimated_cij, entropy_cij, poses, dict_cij


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--wandb_runs', default="wandb_runs.json", type=str, help='runs path')
    parser.add_argument('--dataset', default="mnist", type=str, help='Dataset')
    parser.add_argument('--split', default="test", type=str, help='Dataset split')
    parser.add_argument('--binning', dest='binning', action='store_true')
    parser.add_argument('--no-binning', dest='binning', action='store_false')
    parser.add_argument('--epoch', default='all', type=str, help='Which epoch (default all)')
    parser.add_argument('--entropy_path', default=None, type=str, help='Entropies path (optional)')
    parser.set_defaults(binning=True)
    args = parser.parse_args()
    return args


def get_heatmap_entropy_cij(entropy_cij_nopruning, entropy_cij_pruning, poses_nopruning, poses_pruning, dataset, num_classes):
    heatmap_entropy_coupl_coeff = np.zeros((2, num_classes))

    for i in range(num_classes):
        heatmap_entropy_coupl_coeff[0, i] = np.sum(entropy_cij_nopruning[i]) / entropy_cij_nopruning[i].size
        heatmap_entropy_coupl_coeff[1, i] = np.sum(entropy_cij_pruning[i]) / entropy_cij_pruning[i].size

    return np.sum(heatmap_entropy_coupl_coeff[0])/num_classes, np.sum(heatmap_entropy_coupl_coeff[1])/num_classes
    

def find_checkpoint(folder_path, model):
    files = glob(os.path.join(folder_path, model))
    files.sort(key=os.path.getmtime)
    return files

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#python log_entropies.py --dataset mnist --split test --binning --wandb_run wandb_log_entropies.json
#python log_entropies.py --entropy_path dump/test2.npy
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.entropy_path is not None:
        array = np.load(args.entropy_path)
        print(len(array))
        #intervals1 = np.arange(0, 39, 1)
        intervals = np.arange(0, len(array), 1)
        #intervals = np.concatenate((intervals1, intervals2))
        t = array[intervals,0]
        data1 = array[intervals,1]
        data1 = savitzky_golay(data1, 51, 3) # window size 51, polynomial order 3
        data2 = array[intervals,2]
        data2 = savitzky_golay(data2, 51, 3) # window size 51, polynomial order 3

        fig, ax1 = plt.subplots()

        color = '#b93635'
        ax1.set_xlabel('Pruned parameters (\%)')
        ax1.set_ylabel('Entropy', color="#b93635")
        ax1.plot(100-t, data1, color=color, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor="#b93635")
        #ax1.set_yscale('log2')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = '#3576b9'
        ax2.set_ylabel('Accuracy (\%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(100-t, data2*100, color=color, linewidth=1.5, linestyle = 'dashed')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_size_inches(7, 3.8)
        plt.show()
        fig.savefig("../figures/plot_entropy_pp.pdf", format="pdf", dpi=1200, bbox_inches="tight")
    else:    
        dataset = args.dataset
        binning = args.binning
        runs_path = args.wandb_runs
        split = args.split

        with open(runs_path) as f:
            checkpoints_dict = json.load(f)

        num_seeds = len(checkpoints_dict["datasets"][dataset].keys())
        print(num_seeds)

        if dataset == "tiny-imagenet-200":
            model_name = "ResNetVectorCapsNet"
            classes = 200
            bins = np.linspace(0., 1., num=129, endpoint=True)
        else:
            model_name = "VectorCapsNet"
            classes = 10
            bins = np.linspace(-0.05, 1.05, num=12, endpoint=True)


        key = list(checkpoints_dict["datasets"][dataset].keys())[0]
        run_pruning = api.run("{}/{}/{}".format(wand_settings["entity"], wand_settings["project"],checkpoints_dict["datasets"][dataset][key]["pruning"]["run"]))
        run_name_pruning = run_pruning.name
        print(run_name_pruning)
        checkpoint_pruning = os.path.join("../results/{}/{}/".format(dataset,model_name), checkpoints_dict["experiment_name"], "pruning")

        network_param_non_zero_perc = []
        for i, row in run_pruning.history(keys=["pruning/network_param_non_zero_perc"]).iterrows():
            network_param_non_zero_perc.append(row["pruning/network_param_non_zero_perc"])

        checkpoints_list = find_checkpoint(os.path.join(checkpoint_pruning, run_name_pruning, "checkpoints"), "current*")
        array_to_dump = []
        for index, checkpoint_pruning_path in enumerate(checkpoints_list):
            entropies_pruning = np.zeros(classes)
            config_pruning_path = os.path.join(checkpoint_pruning, run_name_pruning, checkpoints_dict["datasets"][dataset][key]["pruning"]["config"])
            config_pruning = utils.DotDict(json.load(open(config_pruning_path)))
            config_pruning.checkpoint = checkpoint_pruning_path
            config_pruning.binning = binning
            config_pruning.split = split
            config_pruning.augmentation = False
            config_pruning.batch_size = 512

            # PRUNING
            acc_pruning, cij, _, _, _, dict_cij_pruning = main(config_pruning, bins)

            for class_id in range(config_pruning.num_classes): 
                counts_pruning = [dict_cij_pruning[class_id][key] for key in dict_cij_pruning[class_id].keys()]
                entropies_pruning[class_id] += entropy(counts_pruning, base=2)

            entropies_pruning = entropies_pruning / num_seeds
            array_to_dump.append((network_param_non_zero_perc[index], np.mean(entropies_pruning), acc_pruning))
            print("(Param non zero, entropies, acc) {}: ".format((network_param_non_zero_perc[index], np.mean(entropies_pruning), acc_pruning)))
            with open('../dump/entropies.npy', 'wb') as f:
                np.save(f, np.array(array_to_dump))
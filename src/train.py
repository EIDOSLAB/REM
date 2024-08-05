import time
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from tqdm import *
from EIDOSearch.pruning import get_parameters_mask, apply_parameters_mask
from EIDOSearch.regularizers import LOBSTER
from layers.capsule import LinearCaps2d, CapsPrimary2d
import torch.nn as nn
import pynvml

def get_memory_used_MiB(device):
    gpu_index = str(device).split(":")[1]
    print(gpu_index)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = mem_info.used // 1024 ** 2
    total = mem_info.total // 1024 ** 2
    return round(used/total, 5)*100

def train(logging, config, train_loader, model, criterion, optimizer, scheduler, lobster_optimizer, pruning_layers, writer, epoch, device):
    logging.info(
        "-------------------------------------- Training epoch {} --------------------------------------".format(epoch))
    train_vector_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, lobster_optimizer, pruning_layers, writer, epoch,
                             device)


def train_vector_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, lobster_optimizer, pruning_layers, writer, epoch, device):
    num_batches = len(train_loader)
    tot_samples = len(train_loader.sampler)
    loss = 0
    if config.reconstruction is not None:
        margin_loss = 0
        recons_loss = 0
    precision = np.zeros(config.num_classes)
    recall = np.zeros(config.num_classes)
    f1score = np.zeros(config.num_classes)
    correct = 0

    step = epoch * num_batches + num_batches

    model.train()
    if config.model == "ResNetVectorCapsNet":
        model.resnet18.eval()

    start_time = time.time()

    mask_parameters = get_parameters_mask(model, pruning_layers)

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            batch_size = data.size(0)
            # Store the indices for calculating accuracy
            label = target.unsqueeze(0).type(torch.LongTensor)
            global_step = batch_idx + epoch * num_batches
            max_global_step = ((config.epochs + 1) * num_batches)
            relative_step = (1. * global_step) / max_global_step

            # Transform to one-hot indices: [batch_size, 10]
            target = F.one_hot(target, config.num_classes)
            # assert target.size() == torch.Size([batch_size, 10])
            # Use GPU if available
            data, target = data.to(device), target.to(device)

            if config.reconstruction != "None":
                if config.coupl_coeff:
                    class_caps_poses, class_caps_activations, coupling_coefficients, reconstructions = model(data, target)
                else:
                    class_caps_poses, class_caps_activations = model(data)
                c_loss, m_loss, r_loss = criterion(class_caps_activations, target, data, reconstructions,
                                                step=relative_step)

                loss += c_loss.item()
                margin_loss += m_loss.item()
                recons_loss += r_loss.item()
            else:
                if config.coupl_coeff:
                    class_caps_poses, class_caps_activations, coupling_coefficients = model(data)
                    if batch_idx == -1:
                        primary_caps_activations = model.primary_caps_activations # [b, B, h2, w2]
                        primary_caps_activations = primary_caps_activations.view(-1)
                        if writer:
                            writer.add_histogram('primary caps activations', primary_caps_activations, epoch)
                else:
                    class_caps_poses, class_caps_activations = model(data)

                c_loss = criterion(class_caps_activations, target, step=relative_step)

                loss += c_loss.item()

            c_loss.backward()
            if batch_idx == 0:
                perc_used_gpu = get_memory_used_MiB(device)
                wandb.log({'perc_used_gpu': perc_used_gpu}, step=epoch)
            #Train step
            if (batch_idx + 1) % config.iter_size == 0:
                if lobster_optimizer:
                    lobster_optimizer()
                optimizer.step()
                optimizer.zero_grad()
                apply_parameters_mask(model, mask_parameters)

            # Count correct numbers
            # norms: [batch_size, 10, 16]
            # norms = torch.sqrt(torch.sum(class_caps_poses ** 2, dim=2))
            # pred: [batch_size,]
            pred = class_caps_activations.max(1, keepdim=True)[1].type(torch.LongTensor)
            #print(label)
            #print(class_caps_activations)
            correct += pred.eq(label.view_as(pred)).cpu().sum().item()

            # Classification report
            #labels = range(config.num_classes)
            #recall += recall_score(label.view(-1), pred.view(-1), labels=np.unique(pred), average=None)
            #precision += precision_score(label.view(-1), pred.view(-1), labels=np.unique(pred), average=None)
            #f1score += f1_score(label.view(-1), pred.view(-1), labels=np.unique(pred), average=None)

            formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))

            # Print losses
            if batch_idx % config.print_every == 0:
                if config.reconstruction != "None":
                    if config.caps_loss == "margin":
                        logging.info(
                            '\nEpoch: {}    Loss: {:.6f}   Margin loss: {:.6f}   Recons. loss: {:.6f}'.format(
                                formatted_epoch,
                                c_loss.item(),
                                m_loss.item(),
                                r_loss.item()))
                    else:
                        logging.info(
                            '\nEpoch: {}    Loss: {:.6f}   Spread loss: {:.6f}   Recons. loss: {:.6f}'.format(
                                formatted_epoch,
                                c_loss.item(),
                                m_loss.item(),
                                r_loss.item()))
                else:
                    tepoch.set_postfix(loss=c_loss.item())
                    #logging.info('\nEpoch: {}    Loss: {:.6f} '.format(formatted_epoch, c_loss.item()))
            #break
    # Print time elapsed for every epoch
    end_time = time.time()
    logging.info('\nEpoch {} takes {:.0f} seconds for training.'.format(formatted_epoch, end_time - start_time))

    # Log train losses
    loss /= len(train_loader)

    if config.reconstruction != "None":
        margin_loss /= len(train_loader)
        recons_loss /= len(train_loader)

    acc = correct / tot_samples

    # Log classification report
    #recall /= len(train_loader)
    #precision /= len(train_loader)
    #f1score /= len(train_loader)

    # Print classification report
    # logging.info("Training classification report:")
    # for i in range(config.num_classes):
    #     logging.info(
    #         "Class: {} Recall: {:.4f} Precision: {:.4f} F1-Score: {:.4f}".format(i, recall[i], precision[i],
    #                                                                              f1score[i]))

    # Log losses
    if writer: 
        writer.add_scalar('train/loss', c_loss.item(), epoch)
        if config.reconstruction != "None":
            writer.add_scalar('train/margin_loss', m_loss.item(), epoch)
            writer.add_scalar('train/reconstruction_loss', r_loss.item(), epoch)
        writer.add_scalar('train/accuracy', acc, epoch)

    wandb.log({'train/loss': c_loss.item()}, step=epoch)
    if config.reconstruction != "None":
        wandb.log({'train/margin_loss': m_loss.item()}, step=epoch) 
        wandb.log({'train/reconstruction_loss': r_loss.item()}, step=epoch) 
    wandb.log({'train/accuracy': acc}, step=epoch)

    # Print losses
    if config.reconstruction != "None":
        logging.info("Training loss: {:.4f} Margin loss: {:.4f} Recons loss: {:.4f}".format(loss,
                                                                                            margin_loss,
                                                                                            recons_loss))
    else:
        logging.info("Training loss: {:.4f} ".format(loss))

    logging.info("Training accuracy: {}/{} ({:.2f}%)".format(correct, len(train_loader.sampler),
                                                             100. * correct / tot_samples))
    logging.info(
        "Training error: {}/{} ({:.2f}%)".format(tot_samples - correct, tot_samples,
                                                 100. * (1 - correct / tot_samples)))

    # if (config.decay_steps > 0 and global_step % config.decay_steps == 0):
    #     # Update learning rate
    #     scheduler.step()
    #     logging.info('New learning rate: {}'.format(scheduler.get_lr()[0]))

    wandb.log({'lr': scheduler.get_last_lr()[0]}, step=epoch)
    if lobster_optimizer:
        wandb.log({'pruning/lmbda': lobster_optimizer.lmbda}, step=epoch)
    if (config.decay_steps > 0 and epoch % config.decay_steps == 0):
        # Update learning rate
        if (scheduler.get_last_lr()[0] > config.min_lr):
            scheduler.step()
            logging.info('New learning rate: {}'.format(scheduler.get_last_lr()[0]))
        if (lobster_optimizer and lobster_optimizer.lmbda > config.pruning["args"]["min_lmbda"]):
            lobster_optimizer.set_lambda(config.pruning["args"]["max_lmbda"]*(config.decay_rate**(epoch+1)))

    return loss, acc
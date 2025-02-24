from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import sys 
import os

import argparse
import typing
import os
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

import polar_ns.common
import polar_ns.models as models 
from polar_ns.common import LossType, compute_conv_flops
from polar_ns.models.common import SparseGate, Identity
from polar_ns.models.pytorch_lenet5 import LeNet5, lenet5
from polar_ns.prune import prune


def repr_and_saves(seed, cuda = False, log = 'logs', save = 'results', backup_path = 'backup'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not os.path.exists(save):
        os.makedirs(save)
    if backup_path is not None and not os.path.exists(backup_path):
        os.makedirs(backup_path)
    if not os.path.exists(log):
        os.makedirs(log)

def get_loaders(batch_size, test_batch_size): 
    train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data.fashionMNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(2),
                        #transforms.RandomCrop(32),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data.fashionMNIST', train=False, transform=transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader
            
                
def define_optim(model, bn_wd, lr, momentum, weight_decay): 
    if bn_wd:
        no_wd_type = []
    else:
        # do not apply weight decay on bn layers
        no_wd_type = [nn.BatchNorm2d, nn.BatchNorm1d]

    no_wd_params = []  # do not apply weight decay on these parameters
    for module_name, sub_module in model.named_modules():
        for t in no_wd_type:
            if isinstance(sub_module, t):
                for param_name, param in sub_module.named_parameters():
                    no_wd_params.append(param)
                    print(f"No weight decay param: module {module_name} param {param_name}")

    no_wd_params_set = set(no_wd_params)  # apply weight decay on the rest of parameters
    wd_params = []
    for param_name, model_p in model.named_parameters():
        if model_p not in no_wd_params_set:
            wd_params.append(model_p)
            print(f"Weight decay param: parameter name {param_name}")

    optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                                {'params': list(wd_params), 'weight_decay': weight_decay}],
                                lr,
                                momentum=momentum)
    
    return optimizer



def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias


# additional subgradient descent on the sparsity-induced penalty term
def updateBN(config, model):
    if config.get('loss') == LossType.L1_SPARSITY_REGULARIZATION:
        sparsity = config.get('lbd')
        bn_modules = list(filter(lambda m: (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)),
                                 model.named_modules()))
        bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
        for m in bn_modules:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
    else:
        raise NotImplementedError(f"Do not support loss: {config.get('loss')}")


def clamp_bn(model, lower_bound=0, upper_bound=1):
    sparse_modules = list(
        filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))

    for m in sparse_modules:
        m.weight.data.clamp_(lower_bound, upper_bound)


def set_bn_zero(model: nn.Module, threshold=0.0):
    """
    Set bn bias to zero
    Note: The operation is inplace. Parameters of the model will be changed!
    :param model: to set
    :param threshold: set bn bias to zero if corresponding lambda <= threshold
    :return modified model, the number of zero bn channels
    """
    with torch.no_grad():
        mask_length = 0
        for name, sub_module in model.named_modules():
            # only process bn modules
            if not (isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d)):
                continue

            mask = sub_module.weight.detach() <= threshold
            sub_module.weight[mask] = 0.
            sub_module.bias[mask] = 0.

            mask_length += torch.sum(mask).item()

    return model, mask_length


def bn_sparsity(model, loss_type, sparsity, t, alpha,
                flops_weighted: bool, weight_min=None, weight_max=None):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    """
    bn_modules = model.get_sparse_layers()

    if loss_type == LossType.POLARIZATION or loss_type == LossType.L2_POLARIZATION:
        # compute global mean of all sparse vectors
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
        sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

        sparsity_loss = 0.
        if flops_weighted:
            for sub_module in model.modules():
                if isinstance(sub_module, model.building_block):
                    flops_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                    sub_module_sparse_layers = sub_module.get_sparse_modules()

                    for sparse_m, flops_w in zip(sub_module_sparse_layers, flops_weight):
                        # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]
                        flops_w = weight_min + (weight_max - weight_min) * flops_w

                        sparsity_term = t * torch.sum(torch.abs(sparse_m.weight.view(-1))) - torch.sum(
                            torch.abs(sparse_m.weight.view(-1) - alpha * sparse_weights_mean))
                        sparsity_loss += flops_w * sparsity * sparsity_term
            return sparsity_loss
        else:
            for m in bn_modules:
                if loss_type == LossType.POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        torch.abs(m.weight - alpha * sparse_weights_mean))
                elif loss_type == LossType.L2_POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        (m.weight - alpha * sparse_weights_mean) ** 2)
                else:
                    raise ValueError(f"Unexpected loss type: {loss_type}")
                sparsity_loss += sparsity * sparsity_term

            return sparsity_loss
    else:
        raise ValueError()


def train(model, epoch, train_loader, optimizer, lr_scheduler, config, history_score, global_step):
    model.train()
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if config.get('cuda'):
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)

        # logging
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_data += target.data.shape[0]

        if config.get('loss') in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            sparsity_loss = bn_sparsity(model, config.get('loss'), config.get('lbd'),
                                        t=config.get('t'), alpha=config.get('alpha'),
                                        flops_weighted=config.get('flops_weighted'),
                                        weight_max=config.get('weight_max'), weight_min=config.get('weight_min'))
            loss += sparsity_loss
            avg_sparsity_loss += sparsity_loss.data.item()
        loss.backward()
        
        if config.get('loss') in {LossType.L1_SPARSITY_REGULARIZATION}:
            updateBN(config, model)
            
        optimizer.step()
        if config.get('loss') in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            clamp_bn(model, upper_bound=config.get('clamp'))
        global_step += 1
        
        lr_scheduler.step()
        
        if batch_idx % config.get('log_interval') == 0:
            print('Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.data.item()))

    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = float(train_acc) / float(total_data)
    history_score[epoch][3] = avg_sparsity_loss / len(train_loader)
    return history_score, global_step


def test(model, test_loader, config):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if config.get('cuda'):
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if isinstance(output, tuple):
                output, output_aux = output
            test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, backup: bool, backup_path: str, epoch: int, max_backup: int, config):
    state['config'] = config

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    if backup and backup_path is not None:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))

        if max_backup is not None:
            while True:
                # remove redundant backup checkpoints to save space
                checkpoint_match = map(lambda f_name: re.fullmatch("checkpoint_([0-9]+).pth.tar", f_name),
                                       os.listdir(backup_path))
                checkpoint_match = filter(lambda m: m is not None, checkpoint_match)
                checkpoint_id: typing.List[int] = list(map(lambda m: int(m.group(1)), checkpoint_match))
                checkpoint_count = len(checkpoint_id)
                if checkpoint_count > max_backup:
                    min_checkpoint_epoch = min(checkpoint_id)
                    min_checkpoint_path = os.path.join(backup_path,
                                                       'checkpoint_{}.pth.tar'.format(min_checkpoint_epoch))
                    print(f"Too much checkpoints (Max {max_backup}, got {checkpoint_count}).")
                    print(f"Remove file: {min_checkpoint_path}")
                    os.remove(min_checkpoint_path)
                else:
                    break

def fit_model(config): 
    config['cuda'] = not config.get('no_cuda') and torch.cuda.is_available()
    config['loss'] = LossType.from_string(config.get('loss'))
    if not config.get('seed'):
        config['seed'] = random.randint(500, 1000)
        
    
    repr_and_saves(config.get('seed'), config.get('cuda'), config.get('log'), config.get('save'), config.get('backup'))

    train_loader, test_loader = get_loaders(config.get('batch_size'), config.get('test_batch_size'))
    num_classes = config.get('num_classes')
    
    model = lenet5(cfg = config.get('cfg'), bn_init_value=config.get('bn_init_value'), num_classes = num_classes)
        
    optimizer = define_optim(model, config.get('bn_wd'), config.get('lr'), config.get('momentum'), config.get('weight_decay'))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.0, last_epoch=-1, verbose='deprecated')
    
    best_prec1 = 0.0
    global_step = 0
    writer = SummaryWriter(logdir=config.get('log', 'logs'))
    history_score = np.zeros((config.get('epochs'), 6))

    for epoch in range(config.get('start_epoch', 0), config.get('epochs')):
        if config.get('max_epoch') is not None and epoch >= config.get('max_epoch'):
            break
        print("Start epoch {}/{}...".format(epoch, config.get('epochs')))
        
        weights, bias = bn_weights(model)
        for bn_name, bn_weight in weights:
            writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)
        for bn_name, bn_bias in bias:
            writer.add_histogram("bn_bias/" + bn_name, bn_bias, global_step=epoch)
        # visualize conv kernels
        for name, sub_modules in model.named_modules():
            if isinstance(sub_modules, nn.Conv2d):
                writer.add_histogram("conv_kernels/" + name, sub_modules.weight, global_step=epoch)

        
        history_score, global_step = train(model, epoch, train_loader, optimizer, lr_scheduler, config, history_score, global_step)

        prec1 = test(model, test_loader, config)
        history_score[epoch][2] = prec1
        np.savetxt(os.path.join(config.get('save'), 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
        is_best = prec1 > best_prec1  

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=config.get('save', 'results'),
            backup_path=config.get('backup', 'backup'),
            backup=epoch % config.get('backup_freq', 10) == 0,
            epoch=epoch,
            max_backup=config.get('max_backup', 25), 
            config = config
        )
        
        #if counter > patience: 
        # write the tensorboard
        writer.add_scalar("train/average_loss", history_score[epoch][0], epoch)
        writer.add_scalar("train/sparsity_loss", history_score[epoch][3], epoch)
        writer.add_scalar("train/train_acc", history_score[epoch][1], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("val/acc", prec1, epoch)
        writer.add_scalar("val/best_acc", best_prec1, epoch)

    print("Best accuracy: " + str(best_prec1))
    history_score[-1][0] = best_prec1
    np.savetxt(os.path.join(config.get('save'), 'record.txt'), history_score, fmt='%10.5f', delimiter=',')

    writer.close()

    print("Best accuracy: " + str(best_prec1))    
    #print(model.load_state_dict())
    return(best_prec1, model)
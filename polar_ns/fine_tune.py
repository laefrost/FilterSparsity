from __future__ import print_function

import argparse
import os
import random
import re
import shutil
import sys

import numpy as np
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
from polar_ns.models.pytorch_lenet5 import LeNet5
from polar_ns.models.pytorch_lenet5 import LeNet5, lenet5_linear, lenet5



# Training settings
# parser = argparse.ArgumentParser(description='PyTorch CIFAR fine-tuning')
# parser.add_argument('--dataset', type=str, default='cifar10', required=True,
#                     help='training dataset (default: cifar100)')
# parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
#                     help='train with channel sparsity regularization')
# parser.add_argument('--s', type=float, default=0.0001,
#                     help='scale sparse rate (default: 0.0001)')
# parser.add_argument('--refine', type=str, metavar='PATH', required=True,
#                     help='path to the pruned model to be fine tuned')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
#                     help='input batch size for testing (default: 256)')
# parser.add_argument('--epochs', type=int, default=40, metavar='N',
#                     help='number of epochs to train (default: 160)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                     help='learning rate (default: 0.1)')
# parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
#                     help='LR is multiplied by gamma on decay-epoch, number of gammas should be equal to decay-epoch')
# parser.add_argument('--decay-epoch', type=float, nargs='*', default=[0.5, 0.75],
#                     help="the epoch to decay the learning rate (default 0.5, 0.75)")
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, metavar='S', default=None,
#                     help='random seed (default: a random int)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
#                     help='path to save prune model (default: current directory)')
# parser.add_argument('--log', default='./log', type=str, metavar='PATH',
#                     help='path to tensorboard log (default: ./log)')
# parser.add_argument('--arch', default='vgg', type=str,
#                     help='architecture to use')
# parser.add_argument("--expand", action="store_true",
#                     help="use expanded addition in shortcut connections")
# parser.add_argument('--bn-wd', action='store_true',
#                     help='Apply weight decay on BatchNorm layers')
# parser.add_argument('--input-mask', action='store_true',
#                     help='If use input mask in ResNet models.')

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
        no_wd_type = [models.common.SparseGate]
    else:
        # do not apply weight decay on bn layers
        no_wd_type = [models.common.SparseGate, nn.BatchNorm2d, nn.BatchNorm1d]

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


# additional subgradient descent on the sparsity-induced penalty term
def updateBN(config, model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(config['s'] * torch.sign(m.weight.data))  # L1


def train(model, epoch, train_loader, optimizer, lr_scheduler, config, history_score):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if config.get('cuda'):
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)

        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        if config.get('sr'):
            updateBN(config, model)
        optimizer.step()
        lr_scheduler.step()
        if batch_idx % config.get('log_interval', 10) == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = train_acc / float(len(train_loader))
    return history_score


def test(model, test_loader, config):
    model.eval()
    test_loss = 0
    correct = 0
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, config):
    state['config'] = config

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


# def adjust_learning_rate(optimizer, epoch, gammas, schedule):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr
#     assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
#     for (gamma, step) in zip(gammas, schedule):
#         if epoch >= step:
#             lr = lr * gamma
#         else:
#             break
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


def calculate_flops(current_model, num_classes = 10):
    model_ref = lenet5_linear(gate=False)

    current_flops = compute_conv_flops(current_model.cpu())
    ref_flops = compute_conv_flops(model_ref.cpu())
    flops_ratio = current_flops / ref_flops

    print("FLOPs remains {}".format(flops_ratio))
    
    
# --------------------------------------------------------------------------------------------
def fine_tune_model(config): 
    config['cuda'] = not config.get('no_cuda') and torch.cuda.is_available()
    if not config.get('seed'):
        config['seed'] = random.randint(500, 1000)
        
    
    repr_and_saves(config.get('seed'), config.get('cuda'), config.get('log'), config.get('save'), config.get('backup'))

    train_loader, test_loader = get_loaders(config.get('batch_size'), config.get('test_batch_size'))
    num_classes = 10 
    
    if config.get('refine'):
        checkpoint = torch.load(config.get('refine'))
        model = lenet5_linear(gate=False,
                            cfg=checkpoint['cfg'], )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("--refine is required to fine-tune.")
    
    if config.get('resume'):
        if os.path.isfile(config.get('resume')):
            print("=> loading checkpoint '{}'".format(config.get('resume')))
            checkpoint = torch.load(config.get('resume'))
            #args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(config.get('resume'), checkpoint['epoch'], best_prec1))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(config.get('resume')))
    
    
    if config['cuda']:
        model = model.cuda()
    else:
        model = model.cpu()
        
    # test loaded model
    print("Testing the loaded model...")
    test(model, test_loader, config)

    if config.get('evaluate'):
        sys.exit(0)

    calculate_flops(model) 
    optimizer = define_optim(model, config.get('bn_wd'), config.get('lr'), config.get('momentum'), config.get('weight_decay'))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.0, last_epoch=-1, verbose='deprecated')
    
    best_prec1 = 0.0
    writer = SummaryWriter(logdir=config.get('log', 'logs'))
    history_score = np.zeros((config['epochs'], 3))

    for epoch in range(config['epochs']):
        train(model, epoch, train_loader, optimizer, lr_scheduler, config, history_score)
        prec1 = test(model, test_loader, config)
        history_score[epoch][2] = prec1
        np.savetxt(os.path.join(config.get('save'), 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=config.get('save'), config = config)

        # write the tensorboard
        writer.add_scalar("train/average_loss", history_score[epoch][0], epoch)
        writer.add_scalar("train/train_acc", history_score[epoch][1], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("val/acc", prec1, epoch)
        writer.add_scalar("val/best_acc", best_prec1, epoch)

    print("Best accuracy: " + str(best_prec1))
    history_score[-1][0] = best_prec1
    np.savetxt(os.path.join(config.get('save'), 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
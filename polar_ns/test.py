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


def get_loader(test_batch_size): 

    test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data.fashionMNIST', train=False, transform=transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=test_batch_size, shuffle=True)
    
    return test_loader

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

def test_model(config, model = None): 
    test_loader = get_loader(config.get('test_batch_size'))
    num_classes = 10 
    
    if model is None: 
        # TODO: Load checkpoint from path
        pass
    
    acc = test(model, test_loader, config=config)
    return acc
    
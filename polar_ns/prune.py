import argparse
import copy
import os
from typing import Any, Dict

import torch

import  polar_ns.common as common 
from polar_ns.models.common import search_threshold, l1_norm_threshold
from  polar_ns.models.pytorch_lenet5 import LeNet5, lenet5, lenet5_linear
from torchinfo import summary
import numpy as np


def _get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument("--pruning-strategy", type=str,
                        choices=["percent", "fixed", "grad", "search"],
                        help="Pruning strategy", required=True)
    parser.add_argument('--same', action='store_true',
                        help='The model before pruning and after pruning is required to be exactly the same')
    parser.add_argument('--gate', action='store_true',
                        help='Add gate after the BatchNrom layers. Only available for MobileNet v2!')
    parser.add_argument("--prune-mode", type=str, default='default',
                        choices=["multiply", 'default'],
                        help="Pruning mode. Same as `models.common.prune_conv_layer`", )

    return parser


def _check_model_same(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    check if the output is same by same input.
    """
    #print(model2)

    
    model1.eval()
    model2.eval()

    rand_input = torch.rand((8, 1, 32, 32))  # the same input size as CIFAR
    out1 = model1(rand_input)
    out2 = model2(rand_input)
    #print(out1.shape)
    #print(out2.shape)

    diff = out1 - out2
    max_diff = torch.max(diff.abs().view(-1)).item()

    return max_diff


def prune(num_classes: int, sparse_model: torch.nn.Module, pruning_strategy: str, sanity_check: bool,
              prune_mode: str, prune_type: str = 'polarization', l1_norm_ratio=None, l1_norm_cutoff = None):
    """
    :param sparse_model: The model trained with sparsity regularization
    :param pruning_strategy: same as `models.common.search_threshold`
    :param sanity_check: whether do sanity check
    :param prune_mode: same as `models.common.prune_conv_layer`
    :return:
    """
    if isinstance(sparse_model, torch.nn.DataParallel) or isinstance(sparse_model,
                                                                     torch.nn.parallel.DistributedDataParallel):
        sparse_model = sparse_model.module

    # note that pruned model could not do forward pass.
    # need to set channel expand.
    pruned_model = copy.deepcopy(sparse_model)
    pruned_model.cpu()
    #print(pruned_model)
    
    if prune_type == 'polarization':
        pruner = lambda weight: search_threshold(weight, pruning_strategy)
        prune_on = 'factor'
    elif prune_type == 'l1-norm':
        pruner = lambda weight: l1_norm_threshold(weight, ratio=l1_norm_ratio)
        prune_on = 'weight'
    elif prune_type == 'ns':
        # find the threshold
        sparse_layers = pruned_model.get_sparse_layers()
        sparse_weight_concat = np.concatenate([l.weight.data.clone().view(-1).cpu().numpy() for l in sparse_layers])
        sparse_weight_concat = np.abs(sparse_weight_concat)
        sparse_weight_concat = np.sort(sparse_weight_concat)
        thre_index = int(len(sparse_weight_concat) * l1_norm_ratio)
        threshold = sparse_weight_concat[thre_index]
        pruner = lambda weight: threshold
        prune_on = 'factor'
    # only used for ns and polarization 
    elif prune_type == 'l1-norm-cutoff': 
        pruner = lambda weight: l1_norm_cutoff
        prune_on = 'factor'
    else:
        raise ValueError(f"Unsupport prune type: {prune_type}")
    
    # pruned_model.prune_model(pruner=lambda weight: search_threshold(weight, pruning_strategy),
    #                          prune_mode=prune_mode)
    
    pruned_model.prune_model(pruner=pruner,
                             prune_mode=prune_mode,
                             prune_on=prune_on)
    
    print("Pruning finished. cfg:")
    print(pruned_model.config())

    if sanity_check:
         # sanity check: check if pruned model is as same as sparse model
         print("Sanity check: checking if pruned model is as same as sparse model")
         max_diff = _check_model_same(sparse_model, pruned_model)
         print(f"Max diff between Sparse model and Pruned model: {max_diff}\n")

    # load weight to finetuning model
    saved_model = lenet5_linear(gate=False, cfg=pruned_model.config(), )
    #print(saved_model.config())

    pruned_state_dict = {}
    # remove gate param from model
    for param_name, param in pruned_model.state_dict().items():
        if param_name in saved_model.state_dict():
            pruned_state_dict[param_name] = param
        else:
            if "_conv" not in param_name:
                # when the entire block is pruned, the conv parameter will miss, which is expected
                print(f"[WARNING] missing parameter: {param_name}")
                
    #print(pruned_model.state_dict())

    saved_model.load_state_dict(pruned_state_dict)

    if sanity_check:
        print("Sanity check: checking if pruned model is as same as saved model")
        max_diff = _check_model_same(saved_model, pruned_model)
        print(f"Max diff between Saved model and Pruned model: {max_diff}\n")
        assert max_diff < 1e-5, f"Test failed: Max diff should be less than 1e-5, got {max_diff}"

    return saved_model


def main(config):
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #parser = _get_parser()
    #args = parser.parse_args()

    num_classes = 10

    if not os.path.exists(config.get('save')):
        os.makedirs(config.get('save'))

    #print(args)
    #print(f"Current git hash: {common.get_git_id()}")

    if not os.path.isfile(config.get('model')):
        raise ValueError("=> no checkpoint found at '{}'".format(config.get('model')))

    checkpoint: Dict[str, Any] = torch.load(config.get('model'))
    print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")

    #(checkpoint['state_dict'].keys())
    
    # build the sparse model
    sparse_model = lenet5_linear(gate=config.get('gate'))
    
    #print(sparse_model.state_dict().keys())
    sparse_model.load_state_dict(checkpoint['state_dict'])

    saved_model = prune(num_classes=num_classes,
                            sparse_model=sparse_model,
                            pruning_strategy=config.get('pruning_strategy'),
                            sanity_check=True, prune_mode=config.get('prune_mode'), 
                            prune_type = config.get('pruning_type'), l1_norm_ratio=config.get('l1_norm_ratio'), 
                            l1_norm_cutoff=config.get('l1_norm_cutoff'))

    # compute FLOPs
    baseline_flops = common.compute_conv_flops(lenet5_linear(gate=False))
    saved_flops = common.compute_conv_flops(saved_model)

    print(f"Unpruned FLOPs: {baseline_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / baseline_flops:,}")

    # save state_dict
    torch.save({'state_dict': saved_model.state_dict(),
                'cfg': saved_model.config()},
               os.path.join(config.get('save'), f'pruned_{config.get('pruning_strategy')}.pth.tar'))
    
    return baseline_flops, saved_flops

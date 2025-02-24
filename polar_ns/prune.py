import argparse
import copy
import os
from typing import Any, Dict

import torch

import  polar_ns.common as common 
from polar_ns.models.common import search_threshold, l1_norm_threshold
from  polar_ns.models.pytorch_lenet5 import LeNet5, lenet5, SparseConvBlock
from torchinfo import summary
import numpy as np

def _check_model_same(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """
    check if the output is same by same input.
    """
    model1.eval()
    model2.eval()

    rand_input = torch.rand((8, 1, 32, 32))  # the same input size as CIFAR
    out1 = model1(rand_input)
    out2 = model2(rand_input)
    diff = out1 - out2
    max_diff = torch.max(diff.abs().view(-1)).item()

    return max_diff


def prune(sparse_model: torch.nn.Module, pruning_strategy: str, sanity_check: bool, prune_type: str, pruning_hp: float, num_classes: int = 10):
    """
    :param sparse_model: The model trained with sparsity regularization
    :param pruning_strategy: same as `models.common.search_threshold`
    :param sanity_check: whether do sanity check
    :return:
    """
    if isinstance(sparse_model, torch.nn.DataParallel) or isinstance(sparse_model,
                                                                     torch.nn.parallel.DistributedDataParallel):
        sparse_model = sparse_model.module

    pruned_model = copy.deepcopy(sparse_model)
    pruned_model.cpu()
    if prune_type == 'polarization':
        pruner = lambda weight: search_threshold(weight, pruning_strategy)
        prune_on = 'factor'
    elif prune_type == 'l1-norm':
        pruner = lambda weight: l1_norm_threshold(weight, ratio=pruning_hp)
        prune_on = 'weight'
    elif prune_type == 'ns':
        # find the threshold
        sparse_layers = pruned_model.get_sparse_layers()
        sparse_weight_concat = np.concatenate([l.weight.data.clone().view(-1).cpu().numpy() for l in sparse_layers])
        sparse_weight_concat = np.abs(sparse_weight_concat)
        sparse_weight_concat = np.sort(sparse_weight_concat)
        thre_index = int(len(sparse_weight_concat) * pruning_hp)
        threshold = sparse_weight_concat[thre_index]
        pruner = lambda weight: threshold
        prune_on = 'factor'
    elif prune_type == 'pefc': 
        l1_norms = []
        for submodule in pruned_model.modules(): 
            if isinstance(submodule, SparseConvBlock):
                submodule: SparseConvBlock
                conv_weight = submodule.conv.weight.data
                l1_norms_layer = torch.sum(torch.abs(conv_weight), dim=(1, 2, 3))
                l1_norms_layer = l1_norms_layer.cpu().numpy()
                l1_norms = np.append(l1_norms, l1_norms_layer)
        l1_norms = l1_norms[l1_norms != 0]
        sorted_norms = np.sort(l1_norms)
        if len(sorted_norms) == pruning_hp: 
            threshold = max(sorted_norms)
        else: 
            threshold = sorted_norms[pruning_hp]
        pruner = lambda weight: threshold
        prune_on = "l1_norm_weights"
    else:
        raise ValueError(f"Unsupport prune type: {prune_type}")
    
    pruned_model.prune_model(pruner=pruner,
                             prune_on=prune_on)
    
    print("Pruning finished. cfg:")
    print(pruned_model.config())

    if sanity_check:
         # sanity check: check if pruned model is as same as sparse model
         print("Sanity check: checking if pruned model is as same as sparse model")
         max_diff = _check_model_same(sparse_model, pruned_model)
         print(f"Max diff between Sparse model and Pruned model: {max_diff}\n")

    # load weight to finetuning model
    saved_model = lenet5(cfg=pruned_model.config(), num_classes = num_classes)

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


def main(config, sparse_model = None):
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #parser = _get_parser()
    #args = parser.parse_args()
    num_classes = config.get('num_classes')
    if not os.path.exists(config.get('save')):
        os.makedirs(config.get('save'))
        
    if not os.path.isfile(config.get('model')):
        raise ValueError("=> no checkpoint found at '{}'".format(config.get('model')))

    if sparse_model is None: 
        checkpoint: Dict[str, Any] = torch.load(config.get('model'))
        print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")
        sparse_model = lenet5(cfg = config.get('cfg'), num_classes  = num_classes)
        sparse_model.load_state_dict(checkpoint['state_dict'])

    saved_model = prune(sparse_model=sparse_model,
                            pruning_strategy=config.get('pruning_strategy'),
                            sanity_check=True,
                            prune_type = config.get('pruning_type'), pruning_hp=config.get('pruning_hp'), num_classes=num_classes)

    # compute FLOPs
    baseline_flops = common.compute_conv_flops(lenet5(cfg = config.get('cfg'), num_classes = num_classes))
    saved_flops = common.compute_conv_flops(saved_model)

    print(f"Unpruned FLOPs: {baseline_flops:,}")
    print(f"Saved FLOPs: {saved_flops:,}")
    print(f"FLOPs ratio: {saved_flops / baseline_flops:,}")

    # save state_dict
    torch.save({'state_dict': saved_model.state_dict(),
                'cfg': saved_model.config()},
               os.path.join(config.get('save'), f'pruned_{config.get('pruning_strategy')}.pth.tar'))
    
    return baseline_flops, saved_flops, saved_model

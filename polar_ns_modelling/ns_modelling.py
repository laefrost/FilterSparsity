import numpy as np
import pandas as pd
import sys 
import os

# Setup paths
PROJECT_ROOT =  os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from polar_ns.train import fit_model
from polar_ns.prune import main
from polar_ns.fine_tune import fine_tune_model
from polar_ns.test import test_model
import torch.nn as nn

import torch 
from  polar_ns.models.pytorch_lenet5 import LeNet5, lenet5, SparseConvBlock
from polar_ns_modelling.helper import count_nonzero_weights, generate_paths

ratio_min = 0.1
ratio_max = 0.9
ratio_seq_len = 10
ratio_seq = np.linspace(ratio_max, ratio_min, ratio_seq_len)
ratio_seq = np.concatenate([ratio_seq, [0], [0.095, 0.99]])

# config for pp via loss type = "sr"
def load_config_train(lbd, path_sv, path_bckp, path_log):
    config_train = {
    'cfg' : [6, 'A', 16, 'A'],
    'loss' : 'sr', 
    'lbd' : lbd, 
    'epochs' : 3, # Only for testing
    'batch_size' : 256, 
    'test_batch_size' : 256,
    'max_epoch' : None, 
    'lr' : 0.15, 
    'momentum' : 0.9, 
    'weight_decay': 0.0, 
    'num_classes' : 10, 
    'no_cuda': True, 
    'seed' : 1234, 
    'log_interval' : 10,
    'bn_init_value' : 0.5, 
    'flops_weighted' : False,
    'weight-max': None, 
    'weight-min' : None, 
    'bn_wd' : True, 
    'save' : path_sv, 
    'backup' : path_bckp, 
    'log' : path_log
    }
    
    return config_train

def load_config_prune(path_model, path_save, cutoff):
    config_prune = {
    'model' : path_model,
    'batch_size': 256, 
    'test_batch_size' : 256,
    'no_cuda': True, 
    'pruning_type': 'ns', 
    'pruning_strategy' : 'l1_ratio',
    'pruning_hp': cutoff,  
    'save' : path_save,
    'num_classes' : 10
    }
    return config_prune


perf_and_flops = list()
for c in ratio_seq:
    print(c)
    path_sv, path_bckp, path_log = generate_paths(folder = 'ns_test', hp_pruning=c)
    path_model = path_sv + 'model_best.pth.tar'
    path_refine =  path_sv + 'pruned_l1_ratio.pth.tar'
    cfg_train =  load_config_train(0.001, path_sv, path_bckp, path_log)

    #checkpoint = torch.load(path_model)
    #print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")
    #
    #model = lenet5_linear(gate=False)
    #model.load_state_dict(checkpoint['state_dict'])

    if c == 0: 
        cfg_train['loss'] = 'original'
        baseline_flops = 416520
        saved_flops = 416520
    acc, model = fit_model(cfg_train)
    if c > 0: 
        cfg_prune = load_config_prune(path_model, path_sv, c)
        baseline_flops, saved_flops, pruned_model = main(cfg_prune)
        acc = test_model(config=cfg_train, model = pruned_model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(pruned_model.feature) 
    else: 
        acc = test_model(cfg_train, model = model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(model.feature) 
    result = {
        'method' : 'ns_l1_norm_ratio',
        'pruning_hp':  c,  
        'acc': acc, 
        'baseline_flops' : baseline_flops, 
        'remaining_flops' : saved_flops, 
        'nmb_params_before' : nmb_params_before, 
        'nmb_params_after' : nmb_params
    }
    perf_and_flops.append(result)
    
pd.DataFrame(perf_and_flops).to_csv("result_ns_test.csv")
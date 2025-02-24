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

lambda_min = 0.0001
lambda_max = 0.01
lambda_seq_len = 10
lambda_seq =np.linspace(lambda_max, lambda_min, lambda_seq_len)
lambda_seq2 = [0.013, 0.03, 0.035, 0.04, 0.05]
lambda_seq = np.concatenate([lambda_seq, [0], lambda_seq2])

def load_config_train(lbd, path_sv, path_bckp, path_log):
    config_train = {
    'cfg' : [6, 'A', 16, 'A'],
    'loss' : 'zol', 
    'lbd' : lbd, 
    'alpha' : 1, # pp specific
    't' : 1.2,   # pp specific
    'epochs' : 3, # only for testing
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
    'clamp' : 1.0, # pp specific
    'flops_weighted' : False,
    'weight-max': None, 
    'weight-min' : None, 
    'bn_wd' : True, 
    'save' : path_sv, 
    'backup' : path_bckp, 
    'log' : path_log
    }
    
    return config_train

def load_config_prune(path_model, path_save):
    config_prune = {
    'cfg' : [6, 'A', 16, 'A'],
    'model' : path_model,
    'batch_size': 256, 
    'test_batch_size' : 256,
    'no_cuda': True, 
    'pruning_type': 'polarization', 
    'pruning_strategy' : 'grad', # cutoff, fixed
    'pruning_hp' : None, 
    'save' : path_save,
    'num_classes' : 10
    }
    return config_prune


perf_and_flops = list()
for lbd in lambda_seq:
    path_sv, path_bckp, path_log = generate_paths(folder = 'test_pp', hp_pruning=lbd)
    path_model = path_sv + 'model_best.pth.tar'
    path_refine =  path_sv + 'pruned_grad.pth.tar'
    cfg_train =  load_config_train(lbd, path_sv, path_bckp, path_log)
    
    #checkpoint = torch.load(path_model)
    #print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")
    
    #model = lenet5()
    #model.load_state_dict(checkpoint['state_dict'])

    if lbd == 0: 
        cfg_train['loss'] = 'original'
        baseline_flops = 416520
        saved_flops = 416520
    acc, model = fit_model(cfg_train)
    if lbd > 0: 
        cfg_prune = load_config_prune(path_model, path_sv)
        baseline_flops, saved_flops, pruned_model = main(cfg_prune)
        acc = test_model(config=cfg_train, model = pruned_model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(pruned_model.feature)
    else: 
        acc = test_model(cfg_train, model = model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(model.feature) 
    result = {
        'method' : 'pp_grad',
        'pruning_hp' : lbd, 
        'acc': acc, 
        'baseline_flops' : baseline_flops, 
        'remaining_flops' : saved_flops, 
        'nmb_params_before' : nmb_params_before, 
        'nmb_params_after' : nmb_params
    }
    perf_and_flops.append(result)
    
pd.DataFrame(perf_and_flops).to_csv("test_result_pp_grad_test.csv")
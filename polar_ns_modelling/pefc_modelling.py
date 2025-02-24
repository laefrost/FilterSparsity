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
import torch 
import torch.nn as nn
from  polar_ns.models.pytorch_lenet5 import LeNet5, lenet5, SparseConvBlock
from polar_ns_modelling.helper import count_nonzero_weights, generate_paths

pruning_steps = [0, 1, 2, 3, 4]

def load_config_train(lbd, path_sv, path_bckp, path_log):
    config_train = {
    'cfg' : [6, 'A', 16, 'A'],
    'loss' : 'original', 
    'lbd' : lbd, 
    'epochs' : 3, # Only for testing 
    'batch_size' : 256, 
    'test_batch_size' : 256,
    'max_epoch' : None, 
    'lr' : 0.15, 
    'momentum' : 0.9, 
    'weight_decay': 0.0, 
    'num_classes': 10, 
    'no_cuda': True, 
    'seed' : 1234, 
    'log_interval' : 10,
    'bn_init_value' : 0.5,  
    'flops_weighted' : False,
    'weight-max': None, 
    'weight-min' : None, 
    'bn_wd' : False, 
    'save' : path_sv, 
    'backup' : path_bckp, 
    'log' : path_log
    }
    
    return config_train

def load_config_prune(path_model, path_save, pruning_steps):
    config_prune = {
    'cfg' : [6, 'A', 16, 'A'],
    'model' : path_model,
    'batch_size': 256, 
    'test_batch_size' : 256,
    'no_cuda': True, 
    'pruning_type': 'pefc', 
    'pruning_strategy' : 'num_steps',
    'pruning_hp' : pruning_steps, 
    'save' : path_save,
    'num_classes' : 10
    }
    return config_prune


perf_and_flops = list()

path_sv, path_bckp, path_log = generate_paths(folder = 'test_pefc')
path_model = path_sv + 'model_best.pth.tar'
cfg_train =  load_config_train(lbd=0.0, path_sv=path_sv, path_bckp=path_bckp, path_log=path_log)
acc, model = fit_model(cfg_train)

checkpoint = torch.load(path_model)
print(f"=> Loading the model...\n=> Epoch: {checkpoint['epoch']}, Acc.: {checkpoint['best_prec1']}")

model = lenet5(cfg=cfg_train['cfg'], num_classes=cfg_train['num_classes'])
model.load_state_dict(checkpoint['state_dict'])
pruned_model = lenet5(cfg=cfg_train['cfg'], num_classes=cfg_train['num_classes'])
pruned_model.load_state_dict(checkpoint['state_dict'])

for ps in pruning_steps:
    if ps == 0: 
        cfg_train['loss'] = 'original'
        baseline_flops = 416520
        saved_flops = 416520
    #acc, model = fit_model(cfg_train)
    if ps > 0: 
        cfg_prune = load_config_prune(path_model, path_sv, ps-1)
        baseline_flops, saved_flops, pruned_model = main(cfg_prune, pruned_model)
        acc = test_model(config=cfg_train, model = pruned_model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(pruned_model.feature)
    else: 
        acc = test_model(cfg_train, model = model)
        nmb_params_before =  count_nonzero_weights(model.feature)
        nmb_params  = count_nonzero_weights(model.feature) 
    result = {
        'method' : 'pefc',
        'pruning_hp' : ps, 
        'acc': acc, 
        'baseline_flops' : baseline_flops, 
        'remaining_flops' : saved_flops, 
        'nmb_params_before' : nmb_params_before, 
        'nmb_params_after' : nmb_params
    }
    perf_and_flops.append(result)
    
pd.DataFrame(perf_and_flops).to_csv("result_pefc_test.csv")
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

# For NS: use lbd/s/sr as lambda value
lambda_min = 1e-4
#lambda_min = 0.0014
lambda_max = 0.1
lambda_seq_len = 5
lambda_seq = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), lambda_seq_len))
lambda_seq = np.concatenate([lambda_seq, [0]])

# config for pp via loss type = "sr"
def load_config_train(lbd, path_sv, path_bckp, path_log):
    config_train = {
    #'config' : [6, 'A', 16, 'A'],
    'loss' : 'sr', 
    'lbd' : lbd, 
    #'alpha' : 1, 
    #'t' : 1,
    'epochs' : 50, 
    'batch_size' : 256, 
    'test_batch_size' : 256,
    'max_epoch' : None, 
    'lr' : 0.15, 
    'momentum' : 0.9, 
    'weight_decay': 0.0, 
    'resume' : None,
    'no_cuda': True, 
    'seed' : 1234, 
    'log_interval' : 10,
    'bn_init_value' : 0.5, 
    # only used for pp
    'clamp' : 1.0, 
    'gate' : False, 
    'flops_weighted' : False,
    'weight-max': None, 
    'weight-min' : None, 
    'bn_wd' : True, 
    'target-flops' : None, 
    'debug' : False,
    'arch' : 'leNet', 
    'retrain' : False, 
    'save' : path_sv, 
    'backup' : path_bckp, 
    'log' : path_log
    #'save' : './checkpoints_ns/', 
    #'backup' : './backup_ns/', 
    #'log' : './events_ns/'
    }
    
    return config_train

def load_config_prune(path_model, path_save):
    config_prune = {
    'model' : path_model,
    'batch_size': 256, 
    'test_batch_size' : 256,
    'no_cuda': True, 
    # Type of pruning/where to prune on; 'polarization', 'l1-norm', 'ns'
    'pruning_type': 'ns', 
    # How to find prunish threshold; only applied if prune_type == ploarization; ["fixed", "grad", "search"]; for og. PP: grad
    'pruning_strategy' : 'l1_ratio',
    #'Pruning ratio of the L1-Norm'; only applied if prune_type == l1-norm OR ns, default none --> ratio to be pruned
    'l1_norm_ratio': 0.1,  
    # l1_norm_cutoff for other ns and pp threshold finding --> IMPLEMENTED ; CAN BE USED!
    'l1_norm_cutoff': None,
    # None` or `"default"`: default behaviour. The pruning threshold is determined by `sparse_layer
    'prune_mode' : 'default', 
    'save' : path_save,
    'gate' : False
    }
    return config_prune


def load_config_finetune(path_sv, path_bckp, path_log, path_refine):
    config_finetune = {
    #'config' : [6, 'A', 16, 'A'],
    'cuda' : False, 
    'no_cuda' : True, 
    # only relevant if sr TRUE, sr = sparsity regularization --> same as lbd in main
    's' : 0.0001, 
    'sr' : False, # no further sr 
    'epochs' : 50, 
    'batch_size' : 256, 
    'test_batch_size' : 256,
    'max_epoch' : None, 
    'lr' : 0.15, 
    'momentum' : 0.9, 
    'weight_decay': 0.0, 
    'seed' : 1234, 
    'log_interval' : 10,
    'gate' : False, 
    'flops_weighted' : False,
    'bn_wd' : True, 
    'resume' : None,
    'arch' : 'leNet', 
    #'refine' : './checkpoints/pruned_grad.pth.tar',
    'refine': path_refine,
    'save' : path_sv,
    'backup' : path_bckp,
    'log' : path_log}
    return config_finetune

def generate_paths(lbd): 
    save =  './ns_ratio/checkpoints_' + str(lbd) +'/'
    backup = './ns_ratio/backup_'+ str(lbd) +'/'
    log = './ns_ratio/events_' + str(lbd) + '/'
    return save, backup, log

perf_and_flops = list()
for lbd in lambda_seq:
    print(lbd)
    path_sv, path_bckp, path_log = generate_paths(lbd)
    path_model = path_sv + 'model_best.pth.tar'
    path_refine =  path_sv + 'pruned_l1_ratio.pth.tar'
    cfg_train =  load_config_train(lbd, path_sv, path_bckp, path_log)

    if lbd == 0: 
        cfg_train['loss'] = 'original'
        baseline_flops = 416520
        saved_flops = 416520
    acc = fit_model(cfg_train)
    
    if lbd > 0: 
        cfg_prune = load_config_prune(path_model, path_sv)
        baseline_flops, saved_flops = main(cfg_prune)
    
        cfg_finetune = load_config_finetune(path_sv, path_bckp, path_log, path_refine)
        acc = fine_tune_model(cfg_finetune)
    result = {
        'method' : 'ns_l1_norm_ratio',
        'cutoff':  0.1,  
        'lbd' : lbd, 
        'acc': acc, 
        'baseline_flops' : baseline_flops, 
        'remaining_flops' : saved_flops
    }
    perf_and_flops.append(result)
    
pd.DataFrame(perf_and_flops).to_csv("result_ns_l1_ratio.csv")
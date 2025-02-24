# Filter Sparsity 

Implementation of the filter sparsity methods [Only Train Once (OTO)](https://arxiv.org/abs/2107.07467), [Network Slimming (NS)](https://arxiv.org/abs/1708.06519), [Polarization Pruning (PP)](https://github.com/polarizationpruning/PolarizationPruning/blob/master/NIPS2020_PolarizationPruning.pdf) and [PEFC](https://arxiv.org/abs/1608.08710) for LeNet5. The performance is benchmarked using the FashionMNIST dataset.

## Acknowledgement 
The code for NS and Polarization Pruning for LeNet5 was built/adapted based on the [official implementation](https://github.com/polarizationpruning/PolarizationPruning/tree/master) of these methods. 

## Functionality
The scripts for PP, NS and PEFC can be found in ```polar_ns_modelling```. The OTO Notebook (created in Google Colab) can be found in ```oto_modelling```.

### Running PP, NS and PEFC: 
1. Training: Set params in ```load_config_train()```:
    - For PP: ```'loss' : 'zol'```
    - For NS: ```'loss' : 'sr'```
    - For PEFC:  ```'loss' : 'original'```
2. Pruning: Set params in ```load_config_prune()```:
    - For PP:
      ```{python}
      'pruning_type': 'polarization', 
      'pruning_strategy' : 'grad', # 'cutoff' for fixed cutoff (0.05), 'fixed' for the first histogram bin as cutoff```
    - For NS:
    - ```{python}
      'pruning_type': 'ns',
      'pruning_strategy' : 'l1_ratio',
      'pruning_hp': cutoff  # cutoff as sparsity controlling parameter```
    - For PEFC:
      ```{python}
      'pruning_type': 'pefc',
      'pruning_strategy' : 'num_steps',
      'pruning_hp' : pruning_steps   # pruning_steps as sparsity controlling parameter```

import torch 
import torch.nn as nn

def count_nonzero_weights(module: nn.Module) -> int:
    """
    Count the number of non-zero weights in a given module (block).
    """
    nonzero_count = 0

    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Check Conv2D & Linear layers
            nonzero_count += torch.count_nonzero(layer.weight).item()
            print(layer.weight)
    return nonzero_count


def generate_paths(folder: str, hp_pruning = None): 
    if hp_pruning is not None: 
        save =  './' + folder + '/checkpoints_' + str(hp_pruning) +'/'
        backup = './' + folder + '/backup_'+ str(hp_pruning) +'/'
        log = './' + folder + '/events_' + str(hp_pruning) + '/'
    else: 
        save =  './' + folder + '/checkpoints/'
        backup = './' + folder + '/backup/'
        log = './' + folder + '/events/'
    return save, backup, log
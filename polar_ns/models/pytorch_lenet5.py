import torch.nn as nn
import torch.nn.functional as F
import torch

import math
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ['LeNet5']

from polar_ns.models.common import prune_conv_layer, compute_raw_weight, Identity

class SparseConvBlock(nn.Module): 
    def __init__(self, conv : nn.Conv2d, batch_norm : bool, output_channel : int):
        super().__init__()
        
        self.conv = conv
        self.output_channel = output_channel
        
        if batch_norm:
            if isinstance(self.conv, nn.Conv2d):
                self.batch_norm = nn.BatchNorm2d(output_channel)
            elif isinstance(self.conv, nn.Linear):
                self.batch_norm = nn.BatchNorm1d(output_channel)
        else:
            self.batch_norm = Identity()
        self.relu = nn.ReLU(inplace=True)

    @property
    def is_batch_norm(self):
        return not isinstance(self.batch_norm, Identity)

    def forward(self, x):
        conv_out = self.conv(x)
        bn_out = self.batch_norm(conv_out)
        relu_out = self.relu(bn_out)

        return relu_out

    def __repr__(self):
        return f"SparseConvBlock(channel_num={self.output_channel}, " \
               f"bn={self.is_batch_norm})"

    def do_pruning(self, in_channel_mask: np.ndarray, pruner: Callable[[np.ndarray], float], #prune_mode: str, 
                   prune_on: str):
        if not self.is_batch_norm:
            raise ValueError("No sparse layer in the block.")

        out_channel_mask, _ = prune_conv_layer(conv_layer=self.conv,
                                               bn_layer=self.batch_norm if self.is_batch_norm else None,
                                               sparse_layer= self.batch_norm,
                                               in_channel_mask=in_channel_mask,
                                               pruner=pruner,
                                               prune_output_mode="prune",
                                               prune_on=prune_on)
        return out_channel_mask

    def _compute_flops_weight(self, scaling) -> float:
        def scale(raw_value):
            if raw_value is None:
                return None
            return (raw_value - self.raw_weight_min) / (self.raw_weight_max - self.raw_weight_min)

        def identity(raw_value):
            return raw_value

        if scaling:
            scaling_func = scale
        else:
            scaling_func = identity
        return scaling_func(self.raw_flops_weight)

    @property
    def conv_flops_weight(self) -> float:
        """This method is supposed to used in forward pass.
        To use more argument, call `get_conv_flops_weight`."""
        return self.get_conv_flops_weight(update=True, scaling=True)

    def get_conv_flops_weight(self, update: bool, scaling: bool) -> Tuple[float]:
        flops_weight = self._compute_flops_weight(scaling=scaling)

        return (flops_weight,)

    def get_sparse_modules(self) -> Tuple[nn.Module]:
        if self.is_batch_norm:
            return (self.batch_norm,)
        else:
            raise ValueError("No sparse layer available")

    def config(self) -> Tuple[int]:
        if isinstance(self.conv, nn.Conv2d):
            return (self.conv.out_channels,)
        elif isinstance(self.conv, nn.Linear):
            return (self.conv.out_features,)
        else:
            raise ValueError(f"Unsupport conv type: {self.conv}")

class LeNet5(nn.Module): 
    def __init__(self, init_weights=True, cfg = [6, 'A', 16, 'A'], bn_init_value=1, num_classes = 10): 
        super(LeNet5, self).__init__()
        if cfg is None: 
            cfg = [6, 'A', 16, 'A']
        
        self.feature = self.make_layers(cfg, True)
        self.out_dim = self._get_flattened_size(cfg)
        print(self.out_dim)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
             nn.Linear(self.out_dim, 120),  #in_features = 16 x5x5 
             nn.ReLU(),
             nn.Linear(120, 84),
             nn.ReLU(),
             nn.Linear(84, num_classes)
             )
        
        if init_weights:
            self._initialize_weights(bn_init_value)
                    
    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        print(f"LeNet5 make_layers: feature cfg {cfg}")
        for v in cfg:
            if v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=5, stride = 1, bias=False)
                layers.append(SparseConvBlock(conv=conv2d, batch_norm=batch_norm, output_channel=v))
                in_channels = v
        return nn.Sequential(*layers)
    
    
    def _get_flattened_size(self, cfg):
        """Compute the size of the flattened feature map after the convolutional layers."""
        dummy_input = torch.zeros(8, 1, 32, 32)  # Assuming 32x32 grayscale input
        with torch.no_grad():
            output = self.feature(dummy_input)
        self.feature_output_size = output.view(output.size(0), -1).shape[1]
        return self.feature_output_size
        
    
    def forward(self,x): 
        if hasattr(self, 'feature'): 
            x = self.feature(x)
        x = self.flatten(x)
        y = self.classifier(x)
        return y


    def _initialize_weights(self, bn_init_value=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_init_value)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                

    def prune_model(self, pruner: Callable[[np.ndarray], float],
                    prune_on: str) -> None:
        input_mask = np.ones(1)
        for i, submodule in enumerate(self.modules()):
            if isinstance(submodule, SparseConvBlock):
                submodule: SparseConvBlock
                input_mask = submodule.do_pruning(in_channel_mask=input_mask, pruner=pruner, 
                                                  prune_on=prune_on)
            
        input_mask_linear = []
        for i in input_mask: 
            if i == False: 
                input_mask_linear.extend([False] * 25)
            else: 
                input_mask_linear.extend([True] * 25)
        linear_weight: torch.Tensor = self._logit_layer.weight.data.clone()
        
        idx_in = np.squeeze(np.argwhere(np.asarray(input_mask_linear)))
        idx_in_to_zero = np.squeeze(np.argwhere(~np.asarray(input_mask_linear)))
        if len(idx_in.shape) == 0:
            idx_in = np.expand_dims(idx_in, 0)
        linear_weight[:, idx_in_to_zero.tolist()] = 0
        self._logit_layer.weight.data = linear_weight

    @property
    def _logit_layer(self) -> nn.Linear:
        # if self._linear:
        #     return self.classifier[-1]
        # else:
        #     return self.classifier
        return self.classifier[0]

    def get_sparse_layers(self) -> List[nn.Module]:
        sparse_layers: List[nn.Module] = []
        for submodule in self.modules():
            if isinstance(submodule, SparseConvBlock):
                submodule: SparseConvBlock
                if submodule.is_batch_norm:
                    sparse_layers.append(submodule.batch_norm)
                else:
                    raise ValueError("No sparse modules available.")
        return sparse_layers


    def _compute_flops_weight_layerwise(self) -> List[int]:
        vgg_blocks = list(filter(lambda m: isinstance(m, SparseConvBlock), self.modules()))
        flops_weights = []
        for i, block in enumerate(vgg_blocks):
            block: SparseConvBlock
            flops_weight = block.conv.d_flops_out
            if i != len(vgg_blocks) - 1:
                flops_weight += vgg_blocks[i + 1].conv.d_flops_in

            block.raw_flops_weight = flops_weight
            flops_weights.append(flops_weight)

        assert len(flops_weights) == len(vgg_blocks)
        for block in vgg_blocks:
            block.raw_weight_min = min(flops_weights)
            block.raw_weight_max = max(flops_weights)

        return flops_weights

    def compute_flops_weight(self) -> List[Tuple[float]]:
        compute_raw_weight(self, input_size=(32, 32))  # compute d_flops_in and d_flops_out
        self._compute_flops_weight_layerwise()

        conv_flops_weight: List[float] = []
        for submodule in self.modules():
            if isinstance(submodule, SparseConvBlock):
                submodule: SparseConvBlock
                conv_flops_weight.append((submodule.conv_flops_weight,))

        return conv_flops_weight

    @property
    def building_block(self):
        return SparseConvBlock

    def config(self) -> List[int]:
        config = []
        for submodule in self.modules():
            if isinstance(submodule, self.building_block):
                for c in submodule.config():
                    config.append(c)
            elif isinstance(submodule, nn.AvgPool2d):
                config.append('A')

        return config    
    
    
def lenet5(cfg=None, bn_init_value=1, num_classes = 10):
    return LeNet5(init_weights=True, cfg=cfg, bn_init_value=bn_init_value, num_classes=num_classes)


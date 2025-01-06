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

from polar_ns.models.common import SparseGate, prune_conv_layer, compute_raw_weight, BuildingBlock, Identity

class SparseConvBlock(nn.Module): 
    def __init__(self, conv : nn.Conv2d, batch_norm : bool, output_channel : int, gate = True): 
        super().__init__()
        
        self.conv = conv
        self.gate = gate
        self.output_channel = output_channel
        
        if batch_norm:
            if isinstance(self.conv, nn.Conv2d):
                self.batch_norm = nn.BatchNorm2d(output_channel)
            elif isinstance(self.conv, nn.Linear):
                self.batch_norm = nn.BatchNorm1d(output_channel)
        else:
            self.batch_norm = Identity()
        if gate:
            self.sparse_gate = SparseGate(output_channel)
        self.relu = nn.ReLU(inplace=True)


    @property
    def is_batch_norm(self):
        return not isinstance(self.batch_norm, Identity)

    def forward(self, x):
        conv_out = self.conv(x)
        bn_out = self.batch_norm(conv_out)
        if self.gate:
            bn_out = self.sparse_gate(bn_out)
        relu_out = self.relu(bn_out)

        return relu_out

    def __repr__(self):
        return f"SparseConvBlock(channel_num={self.output_channel}, " \
               f"bn={self.is_batch_norm}, " \
               f"gate={self.gate})"

    def do_pruning(self, in_channel_mask: np.ndarray, pruner: Callable[[np.ndarray], float], prune_mode: str, prune_on: str):
        if not self.gate and not self.is_batch_norm:
            raise ValueError("No sparse layer in the block.")

        out_channel_mask, _ = prune_conv_layer(conv_layer=self.conv,
                                               bn_layer=self.batch_norm if self.is_batch_norm else None,
                                               sparse_layer=self.sparse_gate if self.gate else self.batch_norm,
                                               in_channel_mask=in_channel_mask,
                                               pruner=pruner,
                                               prune_output_mode="prune",
                                               prune_mode=prune_mode, 
                                               prune_on=prune_on)

        return out_channel_mask
        pass

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
        if self.gate:
            return (self.sparse_gate,)
        elif self.is_batch_norm:
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
    def __init__(self, gate: bool, dataset='cifar10', depth=19, init_weights=True, cfg = [6, 'A', 16, 'A'], linear=False,
                 bn_init_value=1): 
        super(LeNet5, self).__init__()
        self.gate = gate 
        self._linear = linear
        # self.feature = nn.Sequential(
        #     SparseConvBlock(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0), 
        #                     batch_norm = True, output_channel = 6, gate = gate),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     SparseConvBlock(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), 
        #                     batch_norm = True, output_channel = 16, gate = gate),
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )

        if cfg is None: 
            cfg = [6, 'A', 16, 'A']
            
        # Build feature extractor
        # self.feature = self.make_layers(cfg)
        
        # # Output size from the feature extractor (computed dynamically later)
        # self.feature_output_size = None

        # # Add classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(self._get_flattened_size(cfg), 120),  # First fully connected layer
        #     nn.ReLU(),
        #     nn.Linear(120, 84),  # Second fully connected layer
        #     nn.ReLU(),
        #     nn.Linear(84, 10)  # Output layer
        # )
        
        #if init_weights:
        #    self._initialize_weights()
        
        #self.feature = self.make_layers(cfg, True)
        
        #if linear:
        #    self.feature = self.make_layers(cfg[:-1], True)
        #else:
        #    self.feature = self.make_layers(cfg, True)
        self.feature = self.make_layers(cfg, True)
        
        self.out_dim = self._get_flattened_size(cfg)
        
        #if linear:
        # linear_layer = SparseConvBlock(conv=nn.Linear(out_dim, cfg[-1]), batch_norm=True, output_channel=cfg[-1],
        #                         gate=self.gate)
        # self.classifier = nn.Sequential(
        #     linear_layer,
        #     nn.Linear(cfg[-1], 10)
        #     )
        #else:
        #    self.classifier = nn.Linear(cfg[-1], 10)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
             nn.Linear(self.out_dim, 120),  #in_features = 16 x5x5 
             nn.ReLU(),
             nn.Linear(120, 84),
             nn.ReLU(),
             nn.Linear(84, 10)
             )
        
        if init_weights:
            self._initialize_weights(bn_init_value)

        if self.gate:
            # init SparseGate parameters
            for m in self.modules():
                # avoid the conv be initialized as normal
                # this line should be after the conv initialization
                if isinstance(m, SparseGate):
                    nn.init.constant_(m.weight, 1)
                    
    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        print(f"LeNet5 make_layers: feature cfg {cfg}")
        for v in cfg:
            if v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=5, stride = 1, bias=False)
                layers.append(SparseConvBlock(conv=conv2d, batch_norm=batch_norm, output_channel=v, gate=self.gate))
                in_channels = v
        return nn.Sequential(*layers)
    
    
    def _get_flattened_size(self, cfg):
        """Compute the size of the flattened feature map after the convolutional layers."""
        # Pass a dummy input through the feature extractor to compute output size
        dummy_input = torch.zeros(8, 1, 32, 32)  # Assuming 32x32 grayscale input
        with torch.no_grad():
            output = self.feature(dummy_input)
        #print("Flattend size")
        #print(output.view(output.size(0), -1).shape[1])
        self.feature_output_size = output.view(output.size(0), -1).shape[1]
        return self.feature_output_size
        
    
    def forward(self,x): 
        x=self.feature(x)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        y = self.classifier(x)
        #y = torch.softmax(y)

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

    def prune_model(self, pruner: Callable[[np.ndarray], float], prune_mode: str, prune_on: str) -> None:
        input_mask = np.ones(1)
        for submodule in self.modules():
            if isinstance(submodule, SparseConvBlock):
                submodule: SparseConvBlock
                print(input_mask)
                input_mask = submodule.do_pruning(in_channel_mask=input_mask, pruner=pruner, prune_mode=prune_mode, prune_on=prune_on)

        input_mask_linear = []
        for i in input_mask: 
            if i == False: 
                input_mask_linear.extend([0] * 25)
            else: 
                input_mask_linear.extend([1] * 25)
            
        #print(input_mask_linear)
        # prune the last linear layer
        # TODO: change this 
        linear_weight: torch.Tensor = self._logit_layer.weight.data.clone()
        idx_in = np.squeeze(np.argwhere(np.asarray(input_mask_linear)))
        #print(idx_in)
        if len(idx_in.shape) == 0:
            # expand the single scalar to array
            idx_in = np.expand_dims(idx_in, 0)
        linear_weight = linear_weight[:, idx_in.tolist()]
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
                if self.gate:
                    sparse_layers.append(submodule.sparse_gate)
                else:
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

        # set max_weight and min_weight for each blocks
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


def _test_load_state_dict(net: nn.Module, net_ref: nn.Module):
    conv_list = []
    bn_list = []
    linear_list = []

    conv_idx = 0
    bn_idx = 0
    linear_idx = 0

    for submodule in net.modules():
        if isinstance(submodule, nn.Conv2d):
            conv_list.append(submodule)
        elif isinstance(submodule, nn.BatchNorm2d) or isinstance(submodule, nn.BatchNorm1d):
            bn_list.append(submodule)
        elif isinstance(submodule, nn.Linear):
            linear_list.append(submodule)

    for submodule in net_ref.modules():
        if isinstance(submodule, nn.Conv2d):
            conv_list[conv_idx].load_state_dict(submodule.state_dict())
            conv_idx += 1
        elif isinstance(submodule, nn.BatchNorm2d) or isinstance(submodule, nn.BatchNorm1d):
            bn_list[bn_idx].load_state_dict(submodule.state_dict())
            bn_idx += 1
        elif isinstance(submodule, nn.Linear):
            linear_list[linear_idx].load_state_dict(submodule.state_dict())
            linear_idx += 1


def _check_model_same(net_wo_gate: nn.Module, net_w_gate: nn.Module):
    state_dict = {}
    for key, value in net_w_gate.state_dict().items():
        if key in net_wo_gate.state_dict():
            state_dict[key] = net_wo_gate.state_dict()[key]
        else:
            state_dict[key] = net_w_gate.state_dict()[key]
            print(f"Missing param: {key}")

    net_w_gate.load_state_dict(state_dict)
    
    
def lenet5_linear(gate=False, cfg=None, bn_init_value=1):
    return LeNet5(gate = gate , init_weights=True, cfg = cfg, linear=True, bn_init_value=bn_init_value)


def lenet5(gate, cfg=None, bn_init_value=1):
    return LeNet5(gate = gate, init_weights=True, linear=False, cfg=cfg, bn_init_value=bn_init_value)


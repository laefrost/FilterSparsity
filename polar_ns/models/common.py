import typing
from abc import abstractmethod, ABCMeta
from typing import Union, Callable

import torch
from torch import nn
import numpy as np


class SparseGate(nn.Module):
    def __init__(self, channel_num: int):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=channel_num,
                               out_channels=channel_num,
                               kernel_size=1,
                               groups=channel_num,
                               bias=False)

        # initialize as ones
        nn.init.constant_(self.weight, 1)

        # since the conv layer is already initialized, simply change this variable will change nothing
        # the channel_num should never be changed
        self._channel_num = channel_num

    def forward(self, x):
        input_shape = x.shape
        if len(input_shape) == 2:
            # add two extra dimension
            x = x[:, :, None, None]

        out: torch.Tensor = self._conv(x)

        # restore input shape
        out = out.view(input_shape)

        return out

    @property
    def weight(self) -> torch.nn.Parameter:
        """
        The weight of the gate.

        NOTE: the method does not promise the size of the weight. Make sure do .view(-1)
        before using it as a scaling factor.
        """
        return self._conv.weight

    def clamp(self, upper_bound=1., lower_bound=0.):
        self.weight.data.clamp_(lower_bound, upper_bound)

    def prune(self, idx: np.ndarray) -> None:
        assert len(idx.shape) == 1, f"channel index is supposed to be a 1-d vector, got shape {idx.shape}"

        self.weight.data = self.weight.data[idx.tolist(), :, :, :].clone()
        self._conv.in_channels = len(idx)
        self._conv.out_channels = len(idx)
        self._conv.groups = len(idx)
        self._channel_num = len(idx)
        pass

    def set_ones(self):
        """
        Set the weight to all-one vector.
        This operation actually disable the SparseGate layer.
        After calling the method, this layer is actually an identity mapping.
        """
        self._conv.weight.data = torch.ones_like(self._conv.weight)

    def __repr__(self):
        return f"SparseGate(channel_num={self._channel_num})"
        pass

    def do_pruning(self, mask: np.ndarray):
        idx = np.squeeze(np.argwhere(np.asarray(mask)))
        if len(idx.shape) == 0:
            # expand the single scalar to array
            idx = np.expand_dims(idx, 0)
        elif len(idx.shape) == 1 and idx.shape[0] == 0:
            # nothing left
            raise NotImplementedError("No layer left in input channel")

        self._channel_num = len(idx)

        # prune the conv layer
        self._conv.weight.data = self._conv.weight.data.clone()[idx.tolist(), :, :, :]
        self._conv.in_channels = self._channel_num
        self._conv.out_channels = self._channel_num
        self._conv.groups = self._channel_num

class Identity(torch.nn.Module):
    r"""
    Copy from PyTorch nn.Identity, to replace the nn.Identity in
    PyTorch <= 1.1

    A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


# class BuildingBlock(nn.Module):
#     def do_pruning(self, in_channel_mask: np.ndarray, pruner: Callable[[np.ndarray], float], prune_mode: str):
#         pass

#     def get_conv_flops_weight(self, update: bool, scaling: bool) -> typing.Iterable:
#         pass

#     def get_sparse_modules(self) -> typing.Iterable:
#         pass

#     def config(self) -> typing.Iterable[int]:
#         pass


def l1_norm_threshold(weight: np.ndarray, ratio) -> np.ndarray:
    """return a bool array"""
    assert len(weight.shape) == 4, f"Only support conv weight, got shape: {weight.shape}"
    weight = np.abs(weight)
    out_channels = weight.shape[0]

    l1_norm = np.sum(weight, axis=(1, 2, 3))  # the length is same as output channel number
    num_keep = int(out_channels * (1 - ratio))

    arg_max = np.argsort(l1_norm)
    arg_max_rev = arg_max[::-1][:num_keep]
    mask = np.zeros(out_channels, dtype=np.bool)
    mask[arg_max_rev.tolist()] = True

    return mask


def prune_conv_layer(conv_layer: Union[nn.Conv2d, nn.Linear],
                     bn_layer: nn.BatchNorm2d,
                     sparse_layer: Union[nn.BatchNorm2d],
                     in_channel_mask: np.ndarray,
                     prune_output_mode: str,
                     pruner: Callable[[np.ndarray], float],
                     prune_on="factor") -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    :param prune_output_mode: how to handle the output channel (case insensitive)
        "keep": keep the output channel intact
        "same": keep the output channel as same as input channel
        "prune": prune the output according to bn_layer
    :param pruner: the method to determinate the pruning threshold.
    :param in_channel_mask: a 0-1 vector indicates whether the corresponding channel should be pruned (0) or not (1)
    :param bn_layer: the BatchNorm layer after the convolution layer
    :param conv_layer: the convolution layer to be pruned
    :param sparse_layer: the layer to determine the sparsity. Support BatchNorm2d and SparseGate.
    :param prune_on: 'factor' or 'weight'.
    :return out_channel_mask
    """
    assert isinstance(conv_layer, nn.Conv2d) or isinstance(conv_layer, nn.Linear), f"conv_layer got {conv_layer}"

    assert isinstance(sparse_layer, nn.BatchNorm2d) or \
           isinstance(sparse_layer, nn.BatchNorm1d), \
        f"sparse_layer got {sparse_layer}"

    prune_output_mode = str.lower(prune_output_mode)

    with torch.no_grad():
        conv_weight: torch.Tensor = conv_layer.weight.data.clone()
        out_channel_mask = None
        if in_channel_mask is not None:
            # prune the input channel according to the in_channel_mask
            # convert mask to channel indexes
            idx_in = np.squeeze(np.argwhere(np.asarray(in_channel_mask)))
            if len(idx_in.shape) == 0:
                # expand the single scalar to array
                idx_in = np.expand_dims(idx_in, 0)
                idx_in_to_zero = torch.empty((0), dtype=torch.float32)
            elif len(idx_in.shape) == 1 and idx_in.shape[0] == 0:
                # nothing left, prune the whole block
                out_channel_mask = np.full(conv_layer.out_channels, False)
                idx_in_to_zero = np.squeeze(np.argwhere(~np.asarray(in_channel_mask)))
            else: 
                idx_in_to_zero = np.squeeze(np.argwhere(~np.asarray(in_channel_mask)))
                
            # prune the input of the conv layer
            if isinstance(conv_layer, nn.Conv2d):
                if conv_layer.groups == 1:
                    #conv_weight = conv_weight[:, idx_in.tolist(), :, :]
                    conv_weight[:, idx_in_to_zero.tolist(), :, :] = 0
                else:
                    assert conv_weight.shape[1] == 1, "only works for groups == num_channels"
            elif isinstance(conv_layer, nn.Linear):
                #conv_weight = conv_weight[:, idx_in.tolist()]
                conv_weight[:, idx_in_to_zero.tolist()] = 0
            else:
                raise ValueError(f"unsupported conv layer type: {conv_layer}")

        if out_channel_mask is None: 
            # prune the output channel of the conv layer
            if prune_output_mode == "prune":
                if prune_on == 'factor':
                    sparse_weight: np.ndarray = sparse_layer.weight.view(-1).data.cpu().numpy()
                    # Use abs. weigths: 
                    sparse_weight = np.abs(sparse_weight)
                    output_threshold = pruner(sparse_weight)
                    out_channel_mask: np.ndarray = sparse_weight > output_threshold
                elif prune_on == "weight":
                    out_channel_mask: np.ndarray = pruner(conv_weight.data.cpu().numpy())
                else: # for PEFC
                    l1_norms_layer = torch.sum(torch.abs(conv_weight.data), dim=(1, 2, 3))
                    l1_norms_layer = l1_norms_layer.cpu().numpy()
                    output_threshold = pruner(l1_norms_layer)
                    out_channel_mask: np.ndarray = l1_norms_layer > output_threshold

            elif prune_output_mode == "keep":
                # do not prune the output
                out_channel_mask = np.ones(conv_layer.out_channels)
            elif prune_output_mode == "same":
                # prune the output channel with the input mask
                # keep the conv layer in_channel == out_channel
                out_channel_mask = in_channel_mask
            else:
                raise ValueError(f"invalid prune_output_mode: {prune_output_mode}")

        idx_out: np.ndarray = np.squeeze(np.argwhere(np.asarray(out_channel_mask)))
        idx_to_zero = np.squeeze(np.argwhere(~out_channel_mask))
        if len(idx_out.shape) == 0:
            idx_out = np.expand_dims(idx_out, 0)
            idx_to_zero = np.expand_dims(idx_to_zero, 0)

        if isinstance(conv_layer, nn.Conv2d):
            conv_weight[idx_to_zero, :, :, :] = 0
            conv_layer.weight.data = conv_weight
        elif isinstance(conv_layer, nn.Linear):
            conv_weight[idx_to_zero, :] = 0
            if conv_layer.bias is not None:
                conv_layer.bias[idx_to_zero] = 0
            conv_layer.weight.data = conv_weight
        else:
            raise ValueError(f"unsupported conv layer type: {conv_layer}")

        # prune the bn layer
        if bn_layer is not None:
            bn_layer.weight.data[idx_to_zero] = 0
            bn_layer.bias.data[idx_to_zero] = 0
            bn_layer.running_mean[idx_to_zero] = 0
            bn_layer.running_var[idx_to_zero] = 0

    return out_channel_mask, in_channel_mask


def search_threshold(weight: np.ndarray, alg: str):
    if alg not in ["fixed", "grad", "search", "cutoff"]:
        raise NotImplementedError()

    hist_y, hist_x = np.histogram(weight, bins=100, range=(0, 1))
    if alg == "search":
        raise ValueError(f"Deprecated pruning algorithm: {alg}")
    elif alg == "grad":
        hist_y_diff = np.diff(hist_y)
        for i in range(len(hist_y_diff) - 1):
            if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                threshold = hist_x[i + 1]
                if threshold > 0.2:
                    print(f"WARNING: threshold might be too large: {threshold}")
                return threshold
    elif alg == "fixed":
        return hist_x[1]
    elif alg == "cutoff": 
        return 0.05


def compute_conv_flops_weight(model: nn.Module, building_block, input_size: typing.Tuple[int, int] = (32, 32),
                              cuda=False) -> typing.List[typing.Tuple[int]]:
    """
    compute the conv_flops_weight for the model

    :param building_block: the basic building block for CNN.
     Use BasicBlock for ResNet-56. Use VGGBlock for CIFAR VGG.
    :param input_size: the input spatial size of the network
    """
    if not model.flops_weight_computed or model.input_size is None or model.input_size != input_size:
        # initialization

        # set flag
        model._flops_weight_computed = True
        model.input_size = input_size

        # update when
        # 1. the flops is never be computed
        # 2. the input size is changed
        compute_raw_weight(model, input_size, cuda=cuda)  # compute d_flops_in and d_flops_out

        # the weight is raw weight (without scaling!)
        # now compute the weight min and max for rescaling
        conv_flops_weight_raw: typing.List[typing.Tuple[int, int]] = []
        for submodule in model.modules():
            if isinstance(submodule, building_block):
                #submodule: building_block
                block_raw_weight = submodule.get_conv_flops_weight(update=True, scaling=False)
                conv_flops_weight_raw.append(block_raw_weight)
        # scale weight to [0, 1]
        # set weight_max and weight_min for each building blocks
        weights = []
        for block in conv_flops_weight_raw:
            for i in block:
                weights.append(i)

        max_weights = max(weights)
        min_weights = min(weights)

        # compute the min_weights and max_weights for rescaling
        # for all blocks, the raw_weight_min and the raw_weight_max are same
        for submodule in model.modules():
            if isinstance(submodule, building_block):
                #submodule: building_block
                submodule.raw_weight_min = min_weights
                submodule.raw_weight_max = max_weights

    conv_flops_weight: typing.List[typing.Tuple[int]] = []
    for submodule in model.modules():
        if isinstance(submodule, building_block):
            #submodule: building_block
            conv_flops_weight.append(submodule.conv_flops_weight)

    return conv_flops_weight


def compute_raw_weight(model: nn.Module, input_size: typing.Tuple[int, int], cuda=False):
    """
    compute d_flops_in and d_flops_out for every convolutional layers in the model

    Note: this method needs to do forward pass, which is time-consuming
    """
    model._flops_weight_computed = True

    # register hooks
    hook_handles = []
    for submodule in model.modules():
        if isinstance(submodule, nn.Conv2d):
            hook_handles.append(submodule.register_forward_hook(_conv_raw_weight_hook))
        elif isinstance(submodule, nn.Linear):
            hook_handles.append(submodule.register_forward_hook(_linear_raw_weight_hook))

    # do forward pass to compute the input spatial size for each layer
    random_input = torch.rand(8, 3, *input_size)
    if cuda:
        random_input = random_input.cuda()
        model = model.cuda()
    model(random_input)

    # remove hooks
    for h in hook_handles:
        h.remove()


def _conv_raw_weight_hook(conv: nn.Conv2d, input_data: torch.Tensor, output_data: torch.Tensor):
    """
    a hook to set `d_flops_in` and `d_flops_out` for each convolution layers

    `d_flops_in`: the FLOPs drops when the input channel drops by 1

    `d_flops_out`: the FLOPs drops when the output channel drops by 1
    """
    if conv.groups != 1:
        # for SparseGate and deep-wise layer in MobileNet v2
        # note the `d_flops_in` and `d_flops_out` of SparseGate should NOT be used

        # in MobileNet v2, the groups will change according to the input channel and output channel
        assert conv.groups == conv.in_channels and conv.groups == conv.out_channels

    output_channels, output_height, output_width = output_data[0].size()

    if conv.groups == 1:
        new_conv_groups = conv.groups
    else:
        # the conv_groups will change according to the input channel and output channel
        new_conv_groups = conv.groups - 1

    kernel_ops = conv.kernel_size[0] * conv.kernel_size[1] * (conv.in_channels / new_conv_groups)
    d_kernel_ops_in = conv.kernel_size[0] * conv.kernel_size[1] * (1 / new_conv_groups)

    # flops = kernel_ops * output_channels * output_height * output_width
    if conv.groups == 1:
        # normal conv layer
        conv.d_flops_in = d_kernel_ops_in * output_channels * output_height * output_width
        conv.d_flops_out = kernel_ops * 1 * output_height * output_width
        log_list = [conv.in_channels, conv.out_channels, conv.kernel_size[0], output_height, conv.d_flops_in,
                    conv.d_flops_out]
        log_list = [str(i) for i in log_list]
        print(",".join(log_list))
    else:
        # for deepwise layer
        # this layer will not be pruned, so do not set d_flops_out
        conv.d_flops_in = d_kernel_ops_in * (output_channels - 1) * output_height * output_width


def _linear_raw_weight_hook(linear_layer: nn.Linear, input_data: torch.Tensor, output_data: torch.Tensor):
    input_dim = linear_layer.in_features
    output_dim = linear_layer.out_features

    # flops = linear_layer.weight.nelement()
    assert linear_layer.weight.nelement() == (input_dim * output_dim)
    linear_layer.d_flops_in = (input_dim - 1) * output_dim
    linear_layer.d_flops_out = input_dim * (output_dim - 1)
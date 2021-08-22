import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from typing import NamedTuple

import pickle as pkl

import torchvision.transforms as transforms

import re
battery_pattern = re.compile('(\w{7})-(\w{2}-\w{2}-\w{2}-\w{2})\.pkl\Z')

class BatteryExample(NamedTuple):
    prior_mixtures: torch.Tensor
    actions: torch.Tensor  # Batch * Cycles * ActionDim
    observations: torch.Tensor  # Batch * Cycles * ObsDim
    no: str

    def to(self, device):
        return BatteryExample(prior_mixtures = self.prior_mixtures.to(device),
                              actions=self.actions.to(device),
                              observations=self.observations.to(device),
                              no=self.no)
    def sub_sequence(self, start_idx: int, end_idx = None):
        if end_idx == None:
            end_idx = start_idx
            start_idx = 0
        return BatteryExample(prior_mixtures = self.prior_mixtures[:, start_idx:end_idx],
                              actions=self.actions[:, start_idx:end_idx],
                              observations=self.observations[:, start_idx:end_idx],
                              no=self.no)

def normalize(tensor: torch.Tensor, mean: list, std: list, dim: int = -1, inplace: bool = False) -> torch.Tensor:
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    shape = [1] * len(tensor.shape)
    shape[dim] = -1
    if mean.ndim == 1:
        mean = mean.reshape(shape)
    if std.ndim == 1:
        std = std.reshape(shape)
    tensor.sub_(mean).div_(std)
    return tensor


class BatteryDataset(torch.utils.data.Dataset):
    def __init__(self, cell_list: list, sequence_length: int):
        super().__init__()
        self.cell_list = cell_list
        self.max_len = sequence_length
        self.model_map = {"N190269": torch.tensor([0]), "N190907": torch.tensor([1]), "N192642": torch.tensor([2])}
        self.record_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x[:self.max_len], dtype=torch.float)),
            transforms.Lambda(lambda x: normalize(x, 
            [3.79137357, 571.99807552,  31.55334669,  37.26079976,
            4.19649337, 308.56190824,   6.01797807,  36.43525288,
            3.53299222, 370.40950873, -50.31557134,  38.51907107],
            [2.55015720e-01, 1.78300000e+03, 4.30840000e+01, 6.05629931e+01,
            1.10600000e-01, 1.34200000e+03, 1.21649965e+01, 6.13729525e+01,
            5.19931842e-01, 4.85000000e+02, 2.67549900e+01, 6.00684909e+01])),
        ])
        self.action_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x[:self.max_len], dtype=torch.float)),
            # transforms.Lambda(lambda x: torch.tensor(x[:, 0:1])),
            transforms.Lambda(lambda x: normalize(x, [31.55334669, -50.31557134], [10.9015148 ,  2.60548303])),
        ])
        self.capacity_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x[:self.max_len], dtype=torch.float).unsqueeze(1)),
            transforms.Lambda(lambda x: normalize(x, [47.025], [3.996])),
        ])

    def __getitem__(self, index):
        f = open(self.cell_list[index], "rb")
        data = pkl.load(f)
        assert len(data['state_information']) == len(data['action']) == len(data['capacity'])
        #########################
        # self.max_len = seq_len
        #########################
        result = battery_pattern.search(self.cell_list[index])
        model = result.group(1)
        no = result.group(1) + "-" + result.group(2)
        if model in self.model_map:
            prior_mixtures = self.model_map[model]
        else:
            prior_mixtures = torch.tensor([3])
        records = self.record_transform(data['state_information'])
        actions = self.action_transform(data['action'])
        capacities = self.capacity_transform(data['capacity'])
        observations = torch.cat((records, capacities), dim=-1)
        return BatteryExample(prior_mixtures = prior_mixtures,
                              actions=actions,
                              observations=observations,
                              no=no)

    def __len__(self):
        return len(self.cell_list)

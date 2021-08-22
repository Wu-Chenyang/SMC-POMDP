import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

import pickle as pkl
import numpy as np
import torchvision.transforms as transforms
from typing import NamedTuple

import re
model_pattern = re.compile('(\w{7})-\w{2}-\w{2}-\w{2}-\w{2}\.pkl\Z')


class BatteryExample(NamedTuple):
    prior_mixtures: torch.Tensor
    actions: torch.Tensor # Batch * Cycles * ActionDim
    observations: torch.Tensor # Batch * Cycles * StateDim * Length
    pred_targets: torch.Tensor # Batch * Cycles * 1

    def to(self, device):
        return BatteryExample(prior_mixtures = self.prior_mixtures.to(device),
            actions = self.actions.to(device),
            observations = self.observations.to(device),
            pred_targets = self.pred_targets.to(device))

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
    def __init__(self, cell_list: list, max_sequence_length: int = 4096):
        super().__init__()
        assert max_sequence_length > 0, 'sequence length must be greater than 0'
        self.max_sequence_length = max_sequence_length
        self.cell_list = cell_list
        self.model_map = {"N190269": torch.tensor(0), "N190907": torch.tensor(1), "N192642": torch.tensor(2)}
        self.record_transform = transforms.Compose([
            transforms.Lambda(lambda x: [torch.tensor(item, dtype=torch.float) for item in x]),
            transforms.Lambda(lambda x: [normalize(item, [3.834, -0.025, 28.39, 37.94], [0.322, 33.80, 15.21, 1.178]) for item in x]),
            transforms.Lambda(lambda x: [F.pad(item, (0, 0, 0, self.max_sequence_length - item.shape[-2])) for item in x]),
            transforms.Lambda(lambda x: torch.stack(x, dim=0)),
            transforms.Lambda(lambda x: torch.swapaxes(x, -1, -2)),
        ])
        self.action_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.stack(x, axis=0), dtype=torch.float)),
            transforms.Lambda(lambda x: normalize(x, [31.49, 4.199, -50.03], [6.279, 0.0075, 0.0175])),
        ])
        self.capacity_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(np.stack(x, axis=0), dtype=torch.float)[..., None]),
            transforms.Lambda(lambda x: normalize(x, [47.025], [3.996]))
        ])

    def __getitem__(self, index):
        f = open(self.cell_list[index], "rb")
        data = pkl.load(f)
        assert len(data['state_information']) == len(data['action']) == len(data['capacity'])
        result = model_pattern.search(self.cell_list[index])
        model = result.group(1)
        if model in self.model_map:
            prior_mixtures = F.one_hot(self.model_map[model], num_classes=3).float()
        else:
            prior_mixtures = torch.tensor(3 * [1/3])
        records = self.record_transform(data['state_information'])
        actions = self.action_transform(data['action'])
        capacities = self.capacity_transform(data['capacity'])
        return BatteryExample(prior_mixtures = prior_mixtures, actions=actions, observations=records, pred_targets=capacities)

    def __len__(self):
        return len(self.cell_list)
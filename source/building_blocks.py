import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn.utils import weight_norm

class CNNEncoder(nn.Module):
    def __init__(self, sequence_length: int, input_channel: int, num_channels: list, kernel_sizes: list, strides: list, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            in_channels = input_channel if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [weight_norm(nn.Conv1d(in_channels, out_channels, kernel_sizes[i], stride=strides[i])), nn.LeakyReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
    # Input: Batch * Channel * Length
    # Output: Batch * Channel
        return self.network(x).squeeze(-1)

class CGMM(nn.Module):
    # Conditional Gaussian Mixture Model
    # Yield an Gaussian mixture distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, mixture_num: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num), nn.Softmax(dim=-1))
        self.mean = nn.Linear(hidden_dim, mixture_num * output_dim)
        self.std = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), nn.Softplus())

        self.mixture_num = mixture_num
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        mixtures = self.mixture(features)
        means = self.mean(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
        stds = self.std(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))

        return D.MixtureSameFamily(
            D.Categorical(probs=mixtures),
            D.Independent(D.Normal(means, stds), 1)
        )
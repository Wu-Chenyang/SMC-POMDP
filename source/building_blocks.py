import torch
import torch.nn as nn
import torch.distributions as D
# from torch.nn.utils import weight_norm

class MLP(nn.Module):
    def __init__(self, layer_sizes: list, dropout = 0.3):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.Dropout(dropout), nn.LeakyReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNNEncoder(nn.Module):
    def __init__(self, sequence_length: int, input_channel: int, output_dim: int, num_channels: list, kernel_sizes: list, strides: list):
        super().__init__()
        layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            in_channels = input_channel if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_sizes[i], stride=strides[i]), nn.BatchNorm1d(out_channels), nn.LeakyReLU()]
        self.cnn = nn.Sequential(*layers)
        self.projection_nn = nn.Sequential(nn.Linear(num_channels[-1], output_dim), nn.LeakyReLU())

    def forward(self, x):
    # Input: Batch * Channel * Length
    # Output: Batch * OutputDim
        return self.projection_nn(self.cnn(x).squeeze(-1))

class CGMM(nn.Module):
    # Conditional Gaussian Mixture Model
    # Yield an Gaussian mixture distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, mixture_num: int):
        super().__init__()
        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num))
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
            D.Categorical(logits=mixtures),
            D.Independent(D.Normal(means, stds), 1)
        )

class CGM(nn.Module):
    # Conditional Gaussian Model
    # Yield an Gaussian distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        self.mean = nn.Linear(hidden_dim, output_dim)
        # self.std = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Softplus())
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        means = self.mean(features).reshape((features.shape[:-1] + (self.output_dim,)))
        # stds = self.std(features).reshape((features.shape[:-1] + (self.output_dim,)))
        stds = 1.0
        return D.Independent(D.Normal(means, stds), 1)
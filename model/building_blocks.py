import torch
import torch.nn as nn
import pyro.distributions as D
import numpy as np
import math
from typing import Tuple
# from torch.nn.utils import weight_norm
from utils.util import eps, max_deviation, Softclamp, softclamp

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

class LGM(nn.Module):
    """
        Linear Gaussian Model
    """
    def __init__(self, input_dim: int, output_dim: int, independent: bool = True, identity_covariance: bool = False):
        super().__init__()
        self.output_dim = output_dim
        if identity_covariance or independent:
            self.independent = True

        self.loc = nn.Sequential(nn.Linear(input_dim, output_dim), Softclamp(-max_deviation, max_deviation))
        if identity_covariance:
            self.constant_scale_diag = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        else:
            self.constant_scale_diag = nn.Parameter(torch.full((output_dim,), 0.05))
        if not self.independent:
            self.tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=-1)
            self.diag_indices = np.diag_indices(output_dim)
            self.constant_scale_tril = nn.Parameter(torch.full((output_dim * (output_dim - 1) // 2,), 0.05))
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        locs = self.loc(inputs)
        scale_diags = softclamp(self.constant_scale_diag, eps, max_deviation)
        if self.independent:
            return D.Independent(D.Normal(locs, scale_diags), 1)
        else:
            scales = torch.zeros(inputs.shape[:-1] + (self.output_dim, self.output_dim), device=inputs.device)
            scales[..., self.diag_indices[0], self.diag_indices[1]] = scale_diags
            scales[..., self.tril_indices[0], self.tril_indices[1]] = self.constant_scale_diag
            return D.MultivariateNormal(locs, scale_tril=scales)


class CGMM(nn.Module):
    """
        Conditional Gaussian Mixture Model
        Yield an Gaussian mixture distribution conditioned on the input.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, mixture_num: int, independent: bool = True, identity_covariance: bool = False):
        super().__init__()
        self.mixture_num = mixture_num
        self.output_dim = output_dim
        if identity_covariance or independent:
            self.independent = True

        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        if self.mixture_num > 1:
            self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num), Softclamp(-max_deviation, max_deviation))
        self.loc = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), Softclamp(-max_deviation, max_deviation))
        self.scale_diag = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), Softclamp(eps, max_deviation))
        if identity_covariance:
            torch.nn.init.constant_(self.scale_diag[0].weight, 0.0)
            torch.nn.init.constant_(self.scale_diag[0].bias, 1.0)
            self.scale_diag[0].bias.requires_grad = False
            self.scale_diag[0].weight.requires_grad = False
        else:
            torch.nn.init.constant_(self.scale_diag[0].bias, 0.05)
        if not self.independent:
            self.tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=-1)
            self.diag_indices = np.diag_indices(output_dim)
            self.scale_tril = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim * (output_dim - 1) // 2), Softclamp(-max_deviation, max_deviation))
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        if self.mixture_num > 1:
            mixtures = self.mixture(features)
            locs = self.loc(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
            scale_diags = self.scale_diag(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
            if self.independent:
                return D.MixtureSameFamily(
                    D.Categorical(logits=mixtures),
                    D.Independent(D.Normal(locs, scale_diags), 1)
                )
            else:
                scales = torch.zeros(features.shape[:-1] + (self.mixture_num, self.output_dim, self.output_dim), device=inputs.device)
                scales[..., self.diag_indices[0], self.diag_indices[1]] = scale_diags
                scales[..., self.tril_indices[0], self.tril_indices[1]] = self.scale_tril(features).reshape((features.shape[:-1] + (self.mixture_num, -1)))
                return D.MixtureSameFamily(
                    D.Categorical(logits=mixtures),
                    D.MultivariateNormal(locs, scale_tril=scales)
                )
        else:
            locs = self.loc(features)
            scale_diags = self.scale_diag(features)
            if self.independent:
                return D.Independent(D.Normal(locs, scale_diags), 1)
            else:
                scales = torch.zeros(features.shape[:-1] + (self.output_dim, self.output_dim), device=inputs.device)
                scales[..., self.diag_indices[0], self.diag_indices[1]] = scale_diags
                scales[..., self.tril_indices[0], self.tril_indices[1]] = self.scale_tril(features)
                return D.MultivariateNormal(locs, scale_tril=scales)


class CGGMM(nn.Module):
    """
        Conditional Gated Gaussian Mixture Model. It will return an Gaussian mixture distribution condtion on the inputs.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, mixture_num: int, independent: bool = True, identity_covariance: bool = False):
        super().__init__()
        self.mixture_num = mixture_num
        self.output_dim = output_dim
        self.identity_covariance = identity_covariance
        if identity_covariance or independent:
            self.independent = True

        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        self.linear_gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, mixture_num * output_dim), nn.Sigmoid())
        if self.mixture_num > 1:
            self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num), Softclamp(-max_deviation, max_deviation))

        self.linear_loc = nn.Sequential(nn.Linear(input_dim, mixture_num * output_dim), Softclamp(-max_deviation, max_deviation))
        self.nonlinear_loc = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), Softclamp(-max_deviation, max_deviation))

        self.constant_scale_diag = nn.Parameter(torch.zeros(mixture_num * output_dim), requires_grad=not identity_covariance)
        if identity_covariance:
            self.constant_scale_diag = nn.Parameter(torch.ones(mixture_num * output_dim), requires_grad=False)
        else:
            self.constant_scale_diag = nn.Parameter(torch.full((mixture_num * output_dim,), 0.05))
            self.nonlinear_scale_diag = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), Softclamp(eps, max_deviation))
            torch.nn.init.constant_(self.nonlinear_scale_diag[0].bias, 0.05)
        if not self.independent:
            self.tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=-1)
            self.diag_indices = np.diag_indices(output_dim)
            self.constant_scale_tril = nn.Parameter(torch.full((output_dim * (output_dim - 1) // 2,), 0.05))
            self.nonlinear_scale_tril = nn.Linear(hidden_dim, mixture_num * output_dim * (output_dim - 1) // 2)
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        linear_gates = self.linear_gate(inputs)

        linear_locs = self.linear_loc(inputs)
        nonlinear_locs = self.nonlinear_loc(features)
        locs = linear_gates * linear_locs + (1 - linear_gates) * nonlinear_locs

        if self.identity_covariance:
            scale_diags = self.constant_scale_diag
        else:
            constant_scale_diags = softclamp(self.constant_scale_diag, eps, max_deviation)
            nonlinear_scale_diags = self.nonlinear_scale_diag(features)
            scale_diags = linear_gates * constant_scale_diags + (1 - linear_gates) * nonlinear_scale_diags

        if not self.independent:
            constant_scale_trils = softclamp(self.constant_scale_tril, -max_deviation, max_deviation)
            nonlinear_scale_trils = self.nonlinear_scale_tril(features)
            scale_trils = linear_gates * constant_scale_trils + (1 - linear_gates) * nonlinear_scale_trils

        if self.mixture_num > 1:
            mixtures = self.mixture(features)
            locs = locs.reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
            scale_diags = scale_diags.reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))

            if self.independent:
                return D.MixtureSameFamily(
                    D.Categorical(logits=mixtures),
                    D.Independent(D.Normal(locs, scale_diags), 1)
                )
            else:
                scale_trils = scale_trils.reshape((features.shape[:-1] + (self.mixture_num, -1)))

                scales = torch.zeros(features.shape[:-1] + (self.mixture_num, self.output_dim, self.output_dim), device=inputs.device)
                scales[..., self.diag_indices[0], self.diag_indices[1]] = scale_diags
                scales[..., self.tril_indices[0], self.tril_indices[1]] = scale_trils
                return D.MixtureSameFamily(
                    D.Categorical(logits=mixtures),
                    D.MultivariateNormal(locs, scale_tril=scales)
                )
        else:
            if self.independent:
                return D.Independent(D.Normal(locs, scale_diags), 1)
            else:
                scales = torch.zeros(features.shape[:-1] + (self.output_dim, self.output_dim), device=inputs.device)
                scales[..., self.diag_indices[0], self.diag_indices[1]] = scale_diags
                scales[..., self.tril_indices[0], self.tril_indices[1]] = scale_trils
                return D.MultivariateNormal(locs, scale_tril=scales)
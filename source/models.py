import torch
import torch.nn as nn
import torch.distributions as D

from building_blocks import CNNEncoder, CGMM

from typing import Tuple

class POMDP(nn.Module):
    def __init__(self, state_dim: int = 1, action_dim: int = 1, obs_dim: int = 1,
                trans_mixture_num: int = 2, trans_hidden_dim: int = 50,
                obs_mixture_num: int = 2, obs_hidden_dim: int = 50, device=None):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(state_dim, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.eye(state_dim, device=device), requires_grad=False)
        self.trans_net = CGMM(state_dim + action_dim, state_dim, trans_hidden_dim, trans_mixture_num)
        self.obs_net = CGMM(state_dim + action_dim, obs_dim, obs_hidden_dim, obs_mixture_num)

    def prior(self) -> D.Distribution:
        return D.MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale)
    
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> D.Distribution:
        inputs = torch.cat((state, action), axis=-1)
        return self.trans_net(inputs)

    def observation(self, previous_action: torch.Tensor, state: torch.Tensor) -> D.Distribution:
        inputs = torch.cat((previous_action, state), axis=-1)
        return self.obs_net(inputs)

class BatteryModel(POMDP):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5,
            trans_mixture_num: int = 2, trans_hidden_dim: int = 50,
            obs_mixture_num: int = 2, obs_hidden_dim: int = 50,
            pred_mixture_num: int = 2, pred_hidden_dim: int = 10, device=None,
            # Encoding Net
            obs_channel: int = 4, sequence_length: int = 4096, channels: list = [8] * 5,
            kernel_sizes: list = [4] * 6, strides: list = [4] * 6,
    ):
        super().__init__(state_dim, action_dim, obs_dim, trans_mixture_num, trans_hidden_dim, obs_mixture_num, obs_hidden_dim, device)

        self.prior_mean = nn.Parameter(torch.zeros(3, state_dim, device=device))
        self.prior_scale = nn.Parameter(torch.eye(state_dim, device=device).repeat(3, 1, 1))

        self.encoder = CNNEncoder(sequence_length, obs_channel, channels + [obs_dim], kernel_sizes, strides)

        self.pred_net = CGMM(state_dim, 1, pred_hidden_dim, pred_mixture_num)
    
    def prior(self, prior_mixtures: torch.Tensor) -> D.Distribution:
        return D.MixtureSameFamily(
            D.Categorical(probs=prior_mixtures),
            D.MultivariateNormal(self.prior_mean.expand(prior_mixtures.shape[:-1] + self.prior_mean.shape),
                self.prior_scale.expand(prior_mixtures.shape[:-1] + self.prior_scale.shape))
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
    # Input: Batch * Cycles * Channel * Length
    # Output: Batch * Cycles * EncodedObsDim
        batch_shape = obs.shape[:2]
        feature_shape = obs.shape[2:]
        return self.encoder(obs.reshape((-1,) + feature_shape)).reshape(batch_shape + (-1,))

    def predict(self, hidden_state: torch.Tensor) -> D.Distribution:
        return self.pred_net(hidden_state)
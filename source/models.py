import torch
import torch.nn as nn
import torch.distributions as D

from building_blocks import CGMM, CGM

class POMDP(nn.Module):
    def __init__(self, state_dim: int = 1, action_dim: int = 1, obs_dim: int = 1,
                trans_mixture_num: int = 2, trans_hidden_dim: int = 50, trans_num_hidden_layers: int = 2,
                obs_mixture_num: int = 2, obs_hidden_dim: int = 50, obs_num_hidden_layers: int = 2, device=None):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(state_dim, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(state_dim, device=device), requires_grad=False)
        if trans_mixture_num == 1:
            self.trans_net = CGM(state_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers)
        else:
            self.trans_net = CGMM(state_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num)
        if obs_mixture_num == 1:
            self.obs_net = CGM(state_dim + action_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers)
        else:
            self.obs_net = CGMM(state_dim + action_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers, obs_mixture_num)

    def prior(self) -> D.Distribution:
        return D.Independent(D.Normal(self.prior_mean, self.prior_scale), 1)
    
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> D.Distribution:
        inputs = torch.cat((state, action), axis=-1)
        return self.trans_net(inputs)

    def observation(self, previous_action: torch.Tensor, state: torch.Tensor) -> D.Distribution:
        inputs = torch.cat((previous_action, state), axis=-1)
        return self.obs_net(inputs)

class BatteryModel(POMDP):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5,
            trans_mixture_num: int = 2, trans_hidden_dim: int = 50, trans_num_hidden_layers: int = 2,
            obs_mixture_num: int = 2, obs_hidden_dim: int = 50, obs_num_hidden_layers: int = 2,
            device = None
            # # Encoding Net
            # obs_channel: int = 4, sequence_length: int = 4096, channels: list = [8] * 6,
            # kernel_sizes: list = [4] * 6, strides: list = [4] * 6,
    ):
        super().__init__(state_dim, action_dim, obs_dim,
                        trans_mixture_num, trans_hidden_dim, trans_num_hidden_layers,
                        obs_mixture_num, obs_hidden_dim, obs_num_hidden_layers, device)

        self.prior_mean = nn.Parameter(torch.zeros(4, state_dim, device=device))
        self.prior_scale = nn.Parameter(torch.ones(4, state_dim, device=device))

        # self.encoder = CNNEncoder(sequence_length, obs_channel, obs_dim, channels, kernel_sizes, strides)
    
    def prior(self, prior_mixtures: torch.Tensor) -> D.Distribution:
        means = torch.gather(self.prior_mean, 0, prior_mixtures.expand((-1, self.prior_mean.shape[-1])))
        stds = torch.gather(self.prior_scale, 0, prior_mixtures.expand((-1, self.prior_scale.shape[-1])))
        return D.Independent(D.Normal(means, stds), 1)
    
    # def encode(self, obs: torch.Tensor) -> torch.Tensor:
    # # Input: Batch * Cycles * Channel * Length
    # # Output: Batch * Cycles * EncodedObsDim
    #     batch_shape = obs.shape[:2]
    #     feature_shape = obs.shape[2:]
    #     return self.encoder(obs.reshape((-1,) + feature_shape)).reshape(batch_shape + (-1,))
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as D

import math
from model.models import POMDP, VRNN
from model.building_blocks import CGMM, CGGMM, MLP
from utils.util import max_deviation, softclamp, eps

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
        self.prior_scale = nn.Parameter(torch.full((4, state_dim), 0.1, device=device))

        # self.encoder = CNNEncoder(sequence_length, obs_channel, obs_dim, channels, kernel_sizes, strides)
    
    def prior(self, prior_mixtures: torch.Tensor, **kwargs: dict) -> D.Distribution:
        locs = torch.gather(softclamp(self.prior_mean, -max_deviation, max_deviation), 0, prior_mixtures.expand((-1, self.prior_mean.shape[-1])))
        scales = torch.gather(softclamp(self.prior_scale, eps, max_deviation), 0, prior_mixtures.expand((-1, self.prior_scale.shape[-1])))
        return D.Independent(D.Normal(locs, softclamp(scales, eps, max_deviation)), 1)
    
    # def encode(self, obs: torch.Tensor) -> torch.Tensor:
    # # Input: Batch * Cycles * Channel * Length
    # # Output: Batch * Cycles * EncodedObsDim
    #     batch_shape = obs.shape[:2]
    #     feature_shape = obs.shape[2:]
    #     return self.encoder(obs.reshape((-1,) + feature_shape)).reshape(batch_shape + (-1,))

class VRNNBatteryModel(VRNN):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5,
            trans_mixture_num: int = 2, trans_hidden_dim: int = 50, trans_num_hidden_layers: int = 2,
            obs_mixture_num: int = 2, obs_hidden_dim: int = 50, obs_num_hidden_layers: int = 2,
            proposal_mixture_num: int = 2,  proposal_hidden_dim: int = 50, proposal_num_hidden_layers: int = 2,
            rnn_hidden_dim: int = 50, rnn_num: int = 1,
            state_encoding_dim: int = 10, hist_encoding_dim: int = 10,
            gated_transition: bool = True, independent_trans: bool = True,
            independent_obs: bool = True, identity_obs_covariance: bool = False,
            category_num: int = 4,
            device = None
            # # Encoding Net
            # obs_channel: int = 4, sequence_length: int = 4096, channels: list = [8] * 6,
            # kernel_sizes: list = [4] * 6, strides: list = [4] * 6,
    ):
        super().__init__(state_dim, action_dim, obs_dim,
                        trans_mixture_num, trans_hidden_dim, trans_num_hidden_layers,
                        obs_mixture_num, obs_hidden_dim, obs_num_hidden_layers,
                        proposal_mixture_num,  proposal_hidden_dim, proposal_num_hidden_layers,
                        rnn_hidden_dim, rnn_num,
                        state_encoding_dim, hist_encoding_dim,
                        gated_transition, independent_trans,
                        independent_obs, identity_obs_covariance,
                        device)

        self.prior_mean = nn.Parameter(torch.zeros(4, state_dim, device=device))
        self.prior_scale = nn.Parameter(torch.full((4, state_dim), 0.1, device=device))
        self.category_num = category_num
        self.category_encoder = nn.Linear(category_num, hist_encoding_dim)

        # self.encoder = CNNEncoder(sequence_length, obs_channel, obs_dim, channels, kernel_sizes, strides)
    def prior(self, prior_mixtures, batch_shape, **kwargs):
        locs = torch.gather(softclamp(self.prior_mean, -max_deviation, max_deviation), 0, prior_mixtures.expand((-1, self.prior_mean.shape[-1]))).expand(batch_shape + self.prior_mean.shape[-1:])
        scales = torch.gather(softclamp(self.prior_scale, eps, max_deviation), 0, prior_mixtures.expand((-1, self.prior_scale.shape[-1]))).expand(batch_shape + self.prior_scale.shape[-1:])
        # return D.MultivariateNormal(locs, scale_tril=torch.diag_embed(scales)), kwargs
        return D.Independent(D.Normal(locs, scales), 1), kwargs

    def prior_proposal(self, category: torch.Tensor, batch_shape, **kwargs: dict):
        reshaped_h = kwargs.get('reshaped_h', None)
        if reshaped_h == None:
            raise KeyError("'reshaped_h' not specified")
        category = F.one_hot(category.squeeze(-1), num_classes=self.category_num).float()
        obs_encoding = self.category_encoder(category)
        kwargs['obs_encoding'] = obs_encoding
        return self.proposal_net(torch.cat((obs_encoding, reshaped_h), dim=-1)), kwargs

    def sample_from_prior(model: nn.Module, batch_size: int, seq_len: int, future_actions = None) -> torch.Tensor:
    # sample from model prior
    # Input: current_states None
    #                       BatchDim * StateDim
    #                       StateDim
    #        future_actions None
    #                       Seqlen * BatchDim * ActionDim
    #                       BatchDim * ActionDim 
    #                       ActionDim
        observations = None
        device = next(model.parameters()).device
        batch_shape = (batch_size,)
        long_shape = (model.h0.shape[0],) + len(batch_shape) * (1,) + (model.h0.shape[-1],)
        h = model.h0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        reshaped_h = h.reshape((model.rnn_num, -1, model.rnn_hidden_dim)).transpose(0, 1).reshape(batch_shape + (-1,))
        kwargs = {'hidden_states': h, 'reshaped_h': reshaped_h}
        category = torch.randint(4, (batch_size, 1), device=device)
        prior_distribution, kwargs= model.prior(category, batch_shape, **kwargs)
        current_states = prior_distribution.sample()
        states = current_states.unsqueeze(0)

        if future_actions == None:
            future_actions = torch.zeros(seq_len, batch_size, 2, device=device)
        elif len(future_actions.shape) != 3:
            future_actions = future_actions.expand(seq_len, batch_size, future_actions.shape[-1])
        else:
            assert future_actions.shape[1] == batch_size

        category = F.one_hot(category.squeeze(-1), num_classes=model.category_num).float()
        obs_encoding = model.category_encoder(category)
        kwargs['obs_encoding'] = obs_encoding

        for i in range(seq_len):

            state_encoding = kwargs.get('state_encoding', None)
            if state_encoding is None:
                state_encoding = model.state_encoder(current_states)
            inputs = torch.cat((state_encoding, obs_encoding), dim=-1)
            inputs = inputs.reshape(1, -1, inputs.shape[-1])
            h = h.view(model.rnn_num, -1, model.rnn_hidden_dim)
            output, h = model.rnn(inputs, h)
            h = h.reshape((model.rnn_num,) + batch_shape + (model.rnn_hidden_dim,))
            reshaped_h = output.squeeze(0).reshape(batch_shape + (-1,))
            kwargs['reshaped_h'] = reshaped_h

            current_actions = future_actions[i]
            transition_distribution, kwargs = model.transition(current_states, current_actions, **kwargs)
            current_states = transition_distribution.sample()
            observation_distribution, kwargs = model.observation(current_actions, current_states, **kwargs)
            # current_observations = observation_distribution.sample()
            current_observations = observation_distribution.mean
            obs_encoding = model.ao_encoder(torch.cat((current_actions, current_observations), dim=-1))
            kwargs['obs_encoding'] = obs_encoding

            states = torch.cat((states, current_states.unsqueeze(0)), dim=0)
            if i == 0:
                observations = current_observations.unsqueeze(0)
            else:
                observations = torch.cat((observations, current_observations.unsqueeze(0)), dim=0)

        return states, observations
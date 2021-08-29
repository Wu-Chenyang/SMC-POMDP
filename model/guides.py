
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pyro
import pyro.distributions as D

from typing import Tuple

from copy import deepcopy
from building_blocks import CGMM, MLP, LGM
from itertools import chain
from utils.util import eps, pad_and_reverse


class SmoothingProposal(nn.Module):
    def __init__(self, state_dim: int = 1, proposal_mixture_num: int = 3,
                proposal_hidden_dim: int = 32, proposal_num_hidden_layers: int = 2,
                independent_proposal: bool = True,

                obs_dim: int = 1, with_initial_obs: bool = False,

                rnn_num: int = 2, rnn_hidden_dim: int = 50, bidirectional: bool = False,

                static_state_dim: int = 0, static_info_dim: int = 0,
                action_dim: int = 0
    ):
        super().__init__()
        self.proposal_mixture_num = proposal_mixture_num
        self.proposal_hidden_dim = proposal_hidden_dim
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num = rnn_num
        self.D = 2 if bidirectional else 1

        self.with_static_state = static_state_dim > 0
        self.with_static_info = static_info_dim > 0
        self.with_action = action_dim > 0
        self.with_initial_obs = with_initial_obs

        self.lstm = nn.LSTM(obs_dim + action_dim, rnn_hidden_dim, rnn_num, bidirectional=bidirectional, batch_first=True)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (self.D * rnn_num, 1, rnn_hidden_dim)))
        self.c0 = nn.Parameter(torch.normal(0.0, 0.1, (self.D * rnn_num, 1, rnn_hidden_dim)))
        
        if self.with_static_state:
            if self.with_static_info:
                self.static_info_encoder = nn.Sequential(nn.Linear(static_info_dim, rnn_hidden_dim), nn.Tanh())
            self.static_proposal_net = LGM(rnn_hidden_dim, static_state_dim)
            self.static_state_encoder = nn.Sequential(nn.Linear(static_state_dim, rnn_hidden_dim), nn.Tanh())

        self.prior_proposal_net = LGM(rnn_hidden_dim, state_dim)

        if self.with_initial_obs:
            self.a_0 = nn.Parameter(torch.normal(0.0, 0.1, (1, action_dim)))

        if self.with_action:
            self.action_encoder = nn.Sequential(nn.Linear(action_dim, rnn_hidden_dim), nn.Tanh())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, rnn_hidden_dim), nn.Tanh())
        self.proposal_nn = CGMM(rnn_hidden_dim, state_dim, proposal_hidden_dim, proposal_num_hidden_layers, proposal_mixture_num, independent_proposal)
    
    def encode_future(self, observations: torch.Tensor, actions, lengths):
        batch_size = observations.shape[0]
        h = self.h0.expand(-1, batch_size, -1).contiguous()
        c = self.c0.expand(-1, batch_size, -1).contiguous()
        inputs = torch.cat((observations, actions), dim=-1) if self.with_action else observations
        if self.bidirectional:
            inputs = pack_padded_sequence(torch.flip(inputs, (1,)), lengths, batch_first=True)
            hidden_futures, _ = self.lstm(inputs, (h, c))
            hidden_futures = pad_and_reverse(hidden_futures, lengths)
            hidden_futures = 0.5 * (hidden_futures[:, :, :self.rnn_hidden_dim] + hidden_futures[:, :, self.rnn_hidden_dim:])
        else:
            inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
            hidden_futures, _ = self.lstm(inputs, (h, c))
            hidden_futures, _ = pad_packed_sequence(hidden_futures, batch_first=True)
        return hidden_futures
    
    def guide(self, mini_batch, annealing_factor=1.0):
        observations = mini_batch.observations
        masks = mini_batch.masks
        lengths = mini_batch.lengths
        static_infos = mini_batch.static_infos if self.with_static_info else None

        T_max = mini_batch.observations.size(1)
        batch_size = mini_batch.observations.size(0)
        pyro.module("SmoothingProposal", self)

        if self.with_action:
            actions = mini_batch.actions
            if self.with_initial_obs:
                actions = torch.cat((self.a_0.expand(batch_size, 1, -1), actions), dim=1)
        else:
            actions = None
        hidden_futures = self.encode_future(observations, actions, lengths)

        denominator = 2.0
        current_masks = masks[:, 0:1]
        if self.with_static_state:
            denominator += 1
            infos = hidden_futures[:, 0]
            if self.with_static_info:
                encoded_static_infos = self.static_info_encoder(static_infos)
                infos = 0.5 * (encoded_static_infos + infos)
            with pyro.poutine.scale(None, annealing_factor):
                s_static = pyro.sample('s_static', self.static_proposal_net(infos).mask(current_masks))
            encoded_static_states = self.static_state_encoder(s_static)
            hidden_futures += encoded_static_states.unsqueeze(1)

        with pyro.poutine.scale(None, annealing_factor):
            s_prev = pyro.sample('s_prev', self.prior_proposal_net(0.5 * hidden_futures[:, 0]).mask(current_masks))
        if self.with_initial_obs:
            hidden_futures = hidden_futures[:, 1:]
            masks = masks[:, 1:]

        with pyro.plate("s_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                actions = mini_batch.actions[:, t-1] if self.with_action else None
                current_masks = masks[:, t-1:t]
                encoded_states = self.state_encoder(s_prev)
                infos = (encoded_states + hidden_futures[:, t-1]) / denominator

                with pyro.poutine.scale(None, annealing_factor):
                    s_t = pyro.sample("s_%d" % t,
                                    self.proposal_nn(infos).mask(current_masks))
                s_prev = s_t
        
class NASMCProposal(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5, mixture_num: int = 3, hidden_dim: int = 50, rnn_num: int = 2, num_hidden_layers: int = 2):
        super().__init__()
        self.mixture_num = mixture_num
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.rnn_num = rnn_num
        
        self.lstm = nn.LSTM(obs_dim + state_dim + action_dim, hidden_dim, rnn_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, hidden_dim)))
        self.c0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, hidden_dim)))

        self.dist_nn = CGMM(hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
    
    def reset(self, observations, actions, batch_shape):
        long_shape = (self.h0.shape[0],) + len(batch_shape) * (1,) + (self.h0.shape[-1],)
        h = self.h0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        c = self.c0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        return {'hidden_states': h, 'cell_states': c}

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict):
        h = kwargs.get('hidden_states', None)
        if h == None:
            raise KeyError("'hidden_states' not specified")
        c = kwargs.get('cell_states', None)
        if c == None:
            raise KeyError("'cell_states' not specified")

        batch_shape = observation.shape[:-1]
        rnn_num =  h.shape[0:1]
        h_dim = h.shape[-1:]
        c_dim = c.shape[-1:]
        inputs = torch.cat((previous_state, action, observation), axis=-1)
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        h = h.reshape(rnn_num + (-1,) + h_dim)
        c = c.reshape(rnn_num + (-1,) + c_dim)
        output, (h, c) = self.lstm(inputs, (h,c))
        h = h.reshape(rnn_num + batch_shape + h_dim)
        c = c.reshape(rnn_num + batch_shape + c_dim)
        dists = self.dist_nn(output.reshape(batch_shape + (output.shape[-1],)))
        next_states = dists.sample()
        proposal_log_probs = dists.log_prob(next_states)
        return next_states, proposal_log_probs, proposal_log_probs, {'hidden_states': h, 'cell_states': c}
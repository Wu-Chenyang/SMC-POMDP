
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from typing import Tuple

from copy import deepcopy
from building_blocks import CGMM, MLP, CGM
from itertools import chain
from utils.util import eps


class TASMCProposal(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5, mixture_num: int = 3, hidden_dim: int = 50, rnn_num: int = 2, num_hidden_layers: int = 2):
        super().__init__()
        self.mixture_num = mixture_num
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.rnn_num = rnn_num
        
        self.lstm = nn.LSTM(obs_dim + action_dim, hidden_dim, rnn_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, hidden_dim)))
        self.c0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, hidden_dim)))
        self.hidden_futures = None
        self.future_encodings = None
        
        if mixture_num > 1:
            self.prior_proposal_nn = CGMM(hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
            self.proposal_nn = CGMM(state_dim + hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
        else:
            self.prior_proposal_nn = CGM(hidden_dim, state_dim, hidden_dim, num_hidden_layers)
            self.proposal_nn = CGM(state_dim + hidden_dim, state_dim, hidden_dim, num_hidden_layers)
        self.state_encoder = MLP([state_dim, hidden_dim])
        self.future_encoder = MLP([hidden_dim, hidden_dim])
    
    def proposal_parameters(self):
        return chain(self.proposal_nn.parameters(), (self.h0, self.c0), self.lstm.parameters())
        
    def discriminator_parameters(self):
        return chain(self.state_encoder.parameters(), self.future_encoder.parameters())
    
    def reset(self, observations: torch.Tensor, actions: torch.Tensor, batch_shape):
        batch_size = observations.shape[0]
        h = self.h0.expand(-1, batch_size, -1).contiguous()
        c = self.c0.expand(-1, batch_size, -1).contiguous()
        self.hidden_futures, _ = self.lstm(torch.flip(torch.cat((observations, actions), dim=-1).transpose(0, 1), (0,)), (h, c))
        self.future_encodings = self.future_encoder(self.hidden_futures)
        return {}
    
    def prior_proposal(self, num_particles: int, **kwargs: dict):
        dists, mean_square_scale = self.prior_proposal_nn(self.hidden_futures[-1])
        states = dists.sample((num_particles,))
        state_encoding = self.state_encoder(states)
        future_encoding = self.future_encodings[-1]
        D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        proposal_log_probs = dists.log_prob(states)
        incremental_log_weights = proposal_log_probs - future_log_likelihood
        return states, incremental_log_weights, proposal_log_probs, {}

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict):
        # time_step start from 1
        future = self.hidden_futures[-time_step-1, :, :].unsqueeze(0)
        dists = self.proposal_nn(torch.cat((previous_state, future.expand(previous_state.shape[0], -1, -1)), axis=-1))
        next_states = dists.sample()
        # with torch.no_grad():
        state_encoding = self.state_encoder(previous_state)
        future_encoding = self.future_encodings[-time_step-1]
        D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
        if time_step + 2 <= self.hidden_futures.shape[2]:
            next_state_encoding = self.state_encoder(next_states)
            next_future_encoding = self.future_encodings[-time_step-2]
            D_next = torch.sigmoid((next_state_encoding * next_future_encoding).sum(-1))
        else:
            D_next = torch.full(D.shape, 0.5, device=previous_state.device)
        #############
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        next_future_log_likelihood = torch.log(D_next / (1 - D_next + eps) + eps)
        proposal_log_probs = dists.log_prob(next_states)
        incremental_log_weights = future_log_likelihood - next_future_log_likelihood + proposal_log_probs
        return next_states, incremental_log_weights, proposal_log_probs, {}

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
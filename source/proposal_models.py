
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from typing import Tuple

from copy import deepcopy
from building_blocks import CGMM, MLP
from itertools import chain
eps = 1e-10


class TASMCProposal(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5, mixture_num: int = 3, hidden_dim: int = 50, lstm_num: int = 2, num_hidden_layers: int = 2):
        super().__init__()
        self.mixture_num = mixture_num
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.lstm_num = lstm_num
        
        self.lstm = nn.LSTM(obs_dim + action_dim, hidden_dim, lstm_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (lstm_num, 1, hidden_dim)))
        self.c0 = nn.Parameter(torch.normal(0.0, 0.1, (lstm_num, 1, hidden_dim)))
        self.hidden_futures = None
        self.future_encodings = None
        
        self.proposal_nn = CGMM(state_dim + hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
        self.state_encoder = MLP([state_dim, hidden_dim])
        self.future_encoder = MLP([hidden_dim, hidden_dim])
    
    def proposal_parameters(self):
        return chain(self.proposal_nn.parameters(), (self.h0, self.c0), self.lstm.parameters())
        
    def discriminator_parameters(self):
        return chain(self.state_encoder.parameters(), self.future_encoder.parameters())
    
    def reset(self, observations: torch.Tensor, actions: torch.Tensor, num_particles = 0):
        batch_size = observations.shape[0]
        h = self.h0.expand(-1, batch_size, -1).contiguous()
        c = self.c0.expand(-1, batch_size, -1).contiguous()
        self.hidden_futures, _ = self.lstm(torch.flip(torch.cat((observations, actions), dim=-1).transpose(0, 1), (0,)), (h, c))
        self.future_encodings = self.future_encoder(self.hidden_futures)
        return {}

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        future = self.hidden_futures[-time_step-1, :, :].unsqueeze(0)
        dists = self.proposal_nn(torch.cat((previous_state, future.expand(previous_state.shape[0], -1, -1)), axis=-1))
        next_states = dists.sample()
        with torch.no_grad():
            state_encoding = self.state_encoder(previous_state)
            future_encoding = self.future_encodings[-time_step-1]
            D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
            if time_step + 2 <= self.hidden_futures.shape[2]:
                next_state_encoding = self.state_encoder(next_states)
                next_future_encoding = self.future_encodings[-time_step-2]
                D_next = torch.sigmoid((next_state_encoding * next_future_encoding).sum(-1))
            else:
                D_next = torch.full(D.shape, 0.5, device=previous_state.device)
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        next_future_log_likelihood = torch.log(D_next / (1 - D_next + eps) + eps)
        proposal_log_probs = future_log_likelihood - next_future_log_likelihood + dists.log_prob(next_states)
        return next_states, proposal_log_probs, {}

class NASMCProposal(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5, mixture_num: int = 3, hidden_dim: int = 50, lstm_num: int = 2, num_hidden_layers: int = 2):
        super().__init__()
        self.mixture_num = mixture_num
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.lstm_num = lstm_num
        
        self.lstm = nn.LSTM(obs_dim + state_dim + action_dim, hidden_dim, lstm_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (lstm_num, 1, hidden_dim)))
        self.c0 = nn.Parameter(torch.normal(0.0, 0.1, (lstm_num, 1, hidden_dim)))

        self.dist_nn = CGMM(hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
    
    def reset(self, observations, actions, num_particles):
        batch_shape = (num_particles, observations.shape[0])
        long_shape = (self.h0.shape[0],) + len(batch_shape) * (1,) + (self.h0.shape[-1],)
        h = self.h0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        c = self.c0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        return {'hidden_states': h, 'cell_states': c}

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict) -> Tuple[D.Distribution, Tuple[torch.Tensor, torch.Tensor]]:
        h = kwargs.get('hidden_states', None)
        c = kwargs.get('hidden_states', None)
        if h == None or c == None:
            raise KeyError("'hidden_states' or 'cell_states' not specified")

        batch_shape = observation.shape[:-1]
        lstm_num =  h.shape[0:1]
        h_dim = h.shape[-1:]
        c_dim = c.shape[-1:]
        inputs = torch.cat((previous_state, action, observation), axis=-1)
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        h = h.reshape(lstm_num + (-1,) + h_dim)
        c = c.reshape(lstm_num + (-1,) + c_dim)
        output, (h, c) = self.lstm(inputs, (h,c))
        h = h.reshape(lstm_num + batch_shape + h_dim)
        c = c.reshape(lstm_num + batch_shape + c_dim)
        dists = self.dist_nn(output.reshape(batch_shape + (output.shape[-1],)))
        next_states = dists.sample()
        proposal_log_probs = dists.log_prob(next_states)
        return next_states, proposal_log_probs, {'hidden_states': h, 'cell_states': c}
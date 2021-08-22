import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import math

from utils.util import max_deviation
from model.building_blocks import CGMM, CGM, MLP

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

    def prior(self, batch_shape, **kwargs: dict) -> D.Distribution:
        means = self.prior_mean.clamp(-max_deviation, max_deviation).expand(batch_shape + self.prior_mean.shape)
        scales = F.softplus(self.prior_scale.clamp(-max_deviation, max_deviation), beta=math.log(2.0)).expand(batch_shape + self.prior_scale.shape)
        return D.Independent(D.Normal(means, scales), 1), kwargs
    
    def transition(self, state: torch.Tensor, action: torch.Tensor, **kwargs: dict) -> D.Distribution:
        inputs = torch.cat((state, action), axis=-1)
        dists = self.trans_net(inputs)
        return dists, kwargs

    def observation(self, previous_action: torch.Tensor, state: torch.Tensor, **kwargs: dict) -> D.Distribution:
        inputs = torch.cat((previous_action, state), axis=-1)
        return self.obs_net(inputs), kwargs

class VRNN(nn.Module):
    def __init__(self, state_dim: int = 1, action_dim: int = 1, obs_dim: int = 1,
                trans_mixture_num: int = 2, trans_hidden_dim: int = 50, trans_num_hidden_layers: int = 2,
                obs_mixture_num: int = 2, obs_hidden_dim: int = 50, obs_num_hidden_layers: int = 2,
                proposal_mixture_num: int = 2,  proposal_hidden_dim: int = 50, proposal_num_hidden_layers: int = 2,
                rnn_hidden_dim: int = 50, rnn_num: int = 1,
                device=None):
        super().__init__()
        self.rnn_num = rnn_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.prior_mean = nn.Parameter(torch.zeros(state_dim, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(state_dim, device=device), requires_grad=False)
        self.rnn = nn.GRU(2 * rnn_hidden_dim, rnn_hidden_dim, rnn_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, rnn_hidden_dim)))
        self.state_encoder = MLP([state_dim, rnn_hidden_dim])
        self.ao_encoder = MLP([action_dim + obs_dim, rnn_hidden_dim])
        self.obs_encoder = MLP([obs_dim, rnn_hidden_dim])
        if trans_mixture_num == 1:
            self.trans_net = CGM(rnn_hidden_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers)
        else:
            self.trans_net = CGMM(rnn_hidden_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num)
        if obs_mixture_num == 1:
            self.obs_net = CGM(2 * rnn_hidden_dim + action_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers)
        else:
            self.obs_net = CGMM(2 * rnn_hidden_dim + action_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers, obs_mixture_num)
        if proposal_mixture_num == 1:
            self.proposal_net = CGM(2 * rnn_hidden_dim, state_dim, proposal_hidden_dim, proposal_num_hidden_layers)
        else:
            self.proposal_net = CGMM(2 * rnn_hidden_dim, state_dim, proposal_hidden_dim, proposal_num_hidden_layers, proposal_mixture_num)
    
    def prior(self, batch_shape, **kwargs: dict):
        means = self.prior_mean.clamp(-max_deviation, max_deviation).expand(batch_shape + self.prior_mean.shape)
        scales = F.softplus(self.prior_scale.clamp(-max_deviation, max_deviation), beta=math.log(2.0)).expand(batch_shape + self.prior_scale.shape)
        return D.Independent(D.Normal(means, scales), 1), kwargs


    def transition(self, state, action, **kwargs: dict):
        reshaped_h = kwargs.get('reshaped_h', None)
        if reshaped_h is None:
            raise KeyError("'reshaped_h' not specified")

        return self.trans_net(torch.cat((reshaped_h, action), dim=-1)), kwargs

    def observation(self, previous_action: torch.Tensor, state: torch.Tensor, **kwargs: dict) -> D.Distribution:
        reshaped_h = kwargs.get('reshaped_h', None)
        if reshaped_h == None:
            raise KeyError("'reshaped_h' not specified")

        state_encoding = self.state_encoder(state)
        kwargs['state_encoding'] = state_encoding
        return self.obs_net(torch.cat((previous_action, state_encoding, reshaped_h), axis=-1)), kwargs
    
    def reset(self, observations: torch.Tensor, actions: torch.Tensor, batch_shape):
        long_shape = (self.h0.shape[0],) + len(batch_shape) * (1,) + (self.h0.shape[-1],)
        h = self.h0.reshape(long_shape).expand(-1, *batch_shape, -1).contiguous()
        reshaped_h = h.reshape((self.rnn_num, -1, self.rnn_hidden_dim)).swapaxes(0, 1).reshape(batch_shape + (-1,))
        return {'hidden_states': h, 'reshaped_h': reshaped_h}

    def prior_proposal(self, observation, **kwargs: dict):
        obs_encoding = self.obs_encoder(observation)
        reshaped_h = kwargs.get('reshaped_h', None)
        if reshaped_h == None:
            raise KeyError("'reshaped_h' not specified")
        kwargs['obs_encoding'] = obs_encoding
        return self.proposal_net(torch.cat((obs_encoding, reshaped_h), dim=-1)), kwargs

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict):
        h = kwargs.get('hidden_states', None)
        if h == None:
            raise KeyError("'hidden_states' not specified")
        state_encoding = kwargs.get('state_encoding', None)
        if state_encoding is None:
            state_encoding = self.state_encoder(previous_state)
        obs_encoding = kwargs.get('obs_encoding', None)
        if obs_encoding is None:
            raise KeyError("'obs_encoding' not specified")

        inputs = torch.cat((state_encoding, obs_encoding), dim=-1)
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        h = h.view(self.rnn_num, -1, self.rnn_hidden_dim)
        output, h = self.rnn(inputs, h)
        batch_shape = observation.shape[:-1]
        kwargs['hidden_states'] = h.reshape((self.rnn_num,) + batch_shape + (self.rnn_hidden_dim,))
        reshaped_h = output.squeeze(0).reshape(batch_shape + (-1,))
        kwargs['reshaped_h'] = reshaped_h

        obs_encoding = self.ao_encoder(torch.cat((action, observation), dim=-1))
        kwargs['obs_encoding'] = obs_encoding

        return self.proposal_net(torch.cat((obs_encoding, reshaped_h), dim=-1)), kwargs
    
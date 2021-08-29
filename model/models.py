import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pyro
import pyro.poutine as poutine
import pyro.distributions as D

from utils.util import max_deviation, softclamp, eps
from model.building_blocks import CGMM, CGGMM, MLP, LGM

class HSM(nn.Module):
    """
    A general framework for sequence model with hidden states.
    All methods are designed to work in batch mode and can work independently.

    Attributes
    ----------

    Methods
    -------
    prior(static_info=None)
        Return the prior distribution of HSM along with a status dictionary.
        It can condition on the some static information or simply a unconditioned one.
        If static information is provided, a processed one be included in the status dictionary.
    
    """
    def reset(self):
        """Prints what the animals name is and what sound it makes.

        If the argument `sound` isn't passed in, the default Animal
        sound is used.

        Parameters
        ----------
        sound : str, optional
            The sound the animal makes (default is None)

        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a
            parameter.
        """


class DMM(nn.Module):
    """
        Deep Markov Model
    """
    def __init__(self, state_dim: int = 1, trans_mixture_num: int = 1,
                trans_hidden_dim: int = 32, trans_num_hidden_layers: int = 1,
                gated_transition: bool = True, independent_trans: bool = True,

                obs_dim: int = 1, obs_mixture_num: int = 1,
                obs_hidden_dim: int = 32, obs_num_hidden_layers: int = 1,
                independent_obs: bool = True, identity_obs_covariance: bool = False,
                with_initial_obs: bool = False,

                static_state_dim: int = 0, static_info_dim: int = 0,
                action_dim: int = 0
    ):
        super().__init__()
        self.with_static_state = static_state_dim > 0
        self.with_static_info = static_info_dim > 0
        self.with_action = action_dim > 0
        self.with_initial_obs = with_initial_obs

        if self.with_static_state:
            self.prior_net = LGM(static_state_dim, state_dim)
            if self.with_static_info:
                self.static_prior_net = LGM(static_info_dim, static_state_dim)
            else:
                self.static_state_prior_loc = nn.Parameter(torch.zeros(static_state_dim), requires_grad=False)
                self.static_state_prior_scale_diag = nn.Parameter(torch.ones(static_state_dim), requires_grad=False)
        else:
            self.prior_loc = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
            self.prior_scale_diag = nn.Parameter(torch.ones(state_dim), requires_grad=False)

        if trans_num_hidden_layers == 0:
            self.trans_net = LGM(state_dim + action_dim + static_state_dim, state_dim, independent_trans)
        elif gated_transition:
            self.trans_net = CGGMM(state_dim + action_dim + static_state_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num, independent_trans)
        else:
            self.trans_net = CGMM(state_dim + action_dim + static_state_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num, independent_trans)

        if obs_num_hidden_layers == 0:
            self.obs_net = LGM(state_dim + action_dim + static_state_dim, obs_dim, independent_obs, identity_obs_covariance)
        else:
            self.obs_net = CGMM(state_dim + action_dim + static_state_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers, obs_mixture_num, independent_obs, identity_obs_covariance)
        
        if self.with_initial_obs and self.with_action:
            self.a_0 = nn.Parameter(torch.normal(0.0, 0.1, (1, action_dim)))
        
    def transition(self, state: torch.Tensor, action = None, static_state = None) -> D.Distribution:
        inputs = torch.cat([item for item in (state, action, static_state) if item is not None], axis=-1)
        dists = self.trans_net(inputs)
        return dists

    def observation(self, state: torch.Tensor, previous_action = None, static_state = None) -> D.Distribution:
        inputs = torch.cat([item for item in (state, previous_action, static_state) if item is not None], axis=-1)
        return self.obs_net(inputs)

    def model(self, mini_batch, annealing_factor=1.0):
        T_max = mini_batch.observations.size(1)
        batch_size = mini_batch.observations.size(0)
        masks = mini_batch.masks
        observations = mini_batch.observations

        pyro.module('DMM', self)
        with poutine.scale(None, annealing_factor):
            if self.with_static_state:
                if self.with_static_info:
                    s_static = pyro.sample('s_static',
                                D.Independent(
                                    D.Normal(
                                        self.static_state_prior_loc.expand((batch_size, -1)),
                                        self.static_state_prior_scale_diag.expand((batch_size, -1))
                                    ), 1
                                )
                            )
                else:
                    s_static = pyro.sample("s_static", self.static_prior_net(mini_batch.static_infos))
                s_prev = pyro.sample('s_0', self.prior_net(s_static))
            else:
                s_static = None
                s_prev = pyro.sample('s_0',
                            D.Independent(
                                D.Normal(
                                    self.prior_loc.expand((batch_size, -1)),
                                    self.prior_scale_diag.expand((batch_size, -1))
                                ), 1
                            )
                        )

        if self.with_initial_obs:
            pyro.sample('o_0', self.observation(s_prev,
                                                self.a_0.expand((batch_size, -1)) if self.with_action else None,
                                                s_static)
                                    .mask(masks[:, 0:1]),
                        obs=observations[:, 0])
            masks = masks[:, 1:]
            observations = observations[:, 1:]

        with pyro.plate('s_minibatch', batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                actions = mini_batch.actions[:, t-1] if self.with_action else None
                current_masks = masks[:, t-1:t]
                with poutine.scale(None, annealing_factor):
                    s_t = pyro.sample('s_%d' % t, self.transition(s_prev, actions, s_static)
                                                        .mask(current_masks))
                pyro.sample('o_%d' % t, self.observation(s_t, actions, s_static).mask(current_masks), 
                            obs=observations[:, t-1])
                s_prev = s_t
                


class VRNN(nn.Module):
    def __init__(self, state_dim: int = 1, action_dim: int = 1, obs_dim: int = 1,
                trans_mixture_num: int = 2, trans_hidden_dim: int = 20, trans_num_hidden_layers: int = 2,
                obs_mixture_num: int = 2, obs_hidden_dim: int = 20, obs_num_hidden_layers: int = 2,
                proposal_mixture_num: int = 2,  proposal_hidden_dim: int = 50, proposal_num_hidden_layers: int = 2,
                rnn_hidden_dim: int = 50, rnn_num: int = 1,
                state_encoding_dim: int = 10, hist_encoding_dim: int = 10,
                gated_transition: bool = True, independent_trans: bool = True,
                independent_obs: bool = True, identity_obs_covariance: bool = False,
                device=None):
        super().__init__()
        self.rnn_num = rnn_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.prior_mean = nn.Parameter(torch.zeros(state_dim, device=device), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(state_dim, device=device), requires_grad=False)
        self.rnn = nn.GRU(state_encoding_dim + hist_encoding_dim, rnn_hidden_dim, rnn_num)
        self.h0 = nn.Parameter(torch.normal(0.0, 0.1, (rnn_num, 1, rnn_hidden_dim)))
        self.state_encoder = nn.Linear(state_dim, state_encoding_dim)
        self.ao_encoder = nn.Linear(action_dim + obs_dim, hist_encoding_dim)
        self.obs_encoder = nn.Linear(obs_dim, hist_encoding_dim)
        # if gated_transition:
        #     self.trans_net = CGGMM(rnn_hidden_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num, independent_trans)
        # else:
        #     self.trans_net = CGMM(rnn_hidden_dim + action_dim, state_dim, trans_hidden_dim, trans_num_hidden_layers, trans_mixture_num, independent_trans)
        self.trans_net = LGM(rnn_hidden_dim + action_dim, state_dim, independent_trans)
        self.obs_net = CGMM(state_encoding_dim + rnn_hidden_dim + action_dim, obs_dim, obs_hidden_dim, obs_num_hidden_layers, obs_mixture_num, independent_obs, identity_obs_covariance)
        self.proposal_net = CGMM(hist_encoding_dim + rnn_hidden_dim, state_dim, proposal_hidden_dim, proposal_num_hidden_layers, proposal_mixture_num)
    
    def prior(self, batch_shape, **kwargs: dict):
        locs = softclamp(self.prior_mean, -max_deviation, max_deviation).expand(batch_shape + self.prior_mean.shape)
        scales = softclamp(self.prior_scale, eps, max_deviation).expand(batch_shape + self.prior_scale.shape)
        return D.Independent(D.Normal(locs, scales), 1), kwargs


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
        reshaped_h = h.reshape((self.rnn_num, -1, self.rnn_hidden_dim)).transpose(0, 1).reshape(batch_shape + (-1,))
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
    
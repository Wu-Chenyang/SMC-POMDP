import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as D
from pyro.distributions.kl import kl_divergence

from typing import NamedTuple
import math

def sequence_elbo(proposal: nn.Module, model: nn.Module, batched_data: NamedTuple):
    # Dimensionality Analysis
    #
    # Observations: [batch_size, sequence_length, obs_dim]
    # Weights: [num_particles, batch_size, num_timesteps, 1]
    # Final Weights: [num_particles, batch_size, 1]
    # LSTM State: [num_particles, batch_size, hidden_dim]
    # Current States: [num_particles, batch_size, state_dim]
    # Current Observations: [1, batch_size, obs_dim]
    # Proposal Log Probabilities: [num_particles, batch_size, num_timesteps, 1]
    # Log Likelihood: [batch_size, 1]

    # observations = model.encode(batched_data.observations)
    observations = batched_data.observations
    actions = batched_data.actions
    categories = batched_data.prior_mixtures


    batch_size, seq_len, _ = observations.shape
    batch_shape = (batch_size,)
    kld_loss = torch.tensor([0.0], device=observations.device)
    nll_loss = torch.tensor([0.0], device=observations.device)

    kwargs = proposal.reset(observations, actions, batch_shape)
    proposal_distribution, kwargs = proposal.prior_proposal(categories, batch_shape, **kwargs)
    # proposal_distribution, kwargs = proposal.prior_proposal(observations, batch_shape, **kwargs)
    current_states = proposal_distribution.rsample()

    prior_distribution, kwargs = model.prior(categories, batch_shape, **kwargs)
    kld_loss += torch.mean(kl_divergence(proposal_distribution, prior_distribution))
    # observation_distribution, kwargs= model.observation(current_states, **kwargs)
    # nll_loss -= observation_distribution.log_prob(current_observations)

    for i in range(seq_len):
        current_observations = observations[None, :, i, :]
        current_actions = actions[None, :, i, :]

        previous_states = current_states
        proposal_distribution, kwargs = proposal.transition_proposal(current_states, current_actions, current_observations, i, **kwargs)
        current_states = proposal_distribution.rsample()

        transition_distribution, kwargs= model.transition(previous_states, current_actions, **kwargs)
        kld_loss += torch.mean(kl_divergence(proposal_distribution, transition_distribution))

        observation_distribution, kwargs = model.observation(current_actions, current_states, **kwargs)
        nll_loss -= torch.mean(observation_distribution.log_prob(current_observations))

    kld_loss /= seq_len
    nll_loss /= seq_len
    return kld_loss, nll_loss
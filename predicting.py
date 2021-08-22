import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import NamedTuple
from utils.util import sample_from_prior, batched_index_select, systematic_sampling

def smc_prediction(proposal: nn.Module, model: nn.Module, batched_data: NamedTuple, num_particles: int, pred_steps: int = 1000, pred_batch_size: int = 10) -> torch.Tensor:

    observations = batched_data.observations
    actions = batched_data.actions
    categories = batched_data.prior_mixtures

    batch_size, seq_len, _ = observations.shape
    assert batch_size == 1

    kwargs = proposal.reset(observations, actions, num_particles)

    current_states, incremental_log_weights, proposal_log_probs, kwargs = proposal.prior_proposal(categories, num_particles, **kwargs)
    prior_distribution = model.prior(categories)
    prior_log_probs = prior_distribution.log_prob(current_states)

    model_log_probs = prior_log_probs[..., None, None]
    incremental_log_weights = incremental_log_weights[..., None, None]

    current_logit = model_log_probs - incremental_log_weights
    weights = F.softmax(current_logit, dim=0)
    for i in range(seq_len):
        current_observations = observations[:, i, :].expand(num_particles, -1, -1)
        current_actions = actions[:, i, :].expand(num_particles, -1, -1)

        ancestors = systematic_sampling(weights[..., -1, 0].permute(1, 0)).permute(1, 0)
        resampled_states = batched_index_select(current_states, 0, ancestors, 1)
        kwargs = batched_index_select(kwargs, 1, ancestors, 2)

        current_states, incremental_log_weights, current_proposal_log_probs, kwargs = proposal.transition_proposal(resampled_states.detach(), current_actions, current_observations, i, **kwargs)
        incremental_log_weights = incremental_log_weights[..., None, None]

        transition_distribution = model.transition(resampled_states, current_actions)
        current_transition_log_probs = transition_distribution.log_prob(current_states)

        observation_distribution = model.observation(current_actions, current_states)
        current_observation_log_probs = observation_distribution.log_prob(current_observations)

        current_model_log_probs = (current_transition_log_probs + current_observation_log_probs)[..., None, None]
        current_logit = current_model_log_probs - incremental_log_weights
        current_weights = F.softmax(current_logit, dim=0)

        weights = current_weights

    ancestors = D.Categorical(weights[..., -1, 0].permute(1, 0)).sample((pred_batch_size, ))
    current_states = batched_index_select(current_states, 0, ancestors, 1).squeeze(1)
    states, observations = sample_from_prior(model, pred_batch_size, pred_steps, current_states)
    return observations
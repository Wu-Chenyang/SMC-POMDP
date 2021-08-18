import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import NamedTuple
from smc import batched_index_select, systematic_sampling
from utils import sample_from_prior

def smc_prediction(proposal: nn.Module, model: nn.Module, batched_data: NamedTuple, num_particles: int, pred_steps: int = 1000, pred_batch_size: int = 10) -> torch.Tensor:

    observations = batched_data.observations
    actions = batched_data.actions

    batch_size, seq_len, _ = observations.shape
    assert batch_size == 1

    proposal_args = proposal.reset(observations, actions, num_particles)

    prior_distribution = model.prior(batched_data.prior_mixtures)
    current_states = prior_distribution.rsample((num_particles,))
    weights = torch.ones((num_particles, batch_size, 1, 1), device=observations.device) / num_particles

    for i in range(seq_len):
        current_observations = observations[:, i, :].expand(num_particles, -1, -1)
        current_actions = actions[:, i, :].expand(num_particles, -1, -1)

        if i == 0:
            resampled_states = current_states
        else:
            ancestors = systematic_sampling(weights[..., -1, 0].permute(1, 0)).permute(1, 0)
            resampled_states = batched_index_select(current_states, 0, ancestors, 1)
            proposal_args = batched_index_select(proposal_args, 1, ancestors, 2)

        current_states, current_proposal_log_probs, proposal_args = proposal.transition_proposal(resampled_states.detach(), current_actions, current_observations, i, **proposal_args)
        current_proposal_log_probs = current_proposal_log_probs[..., None, None]

        transition_distribution = model.transition(resampled_states, current_actions)
        current_transition_log_probs = transition_distribution.log_prob(current_states)

        observation_distribution = model.observation(current_actions, current_states)
        current_observation_log_probs = observation_distribution.log_prob(current_observations)

        current_model_log_probs = (current_transition_log_probs + current_observation_log_probs)[..., None, None]
        current_logit = current_model_log_probs - current_proposal_log_probs.detach()
        current_weights = F.softmax(current_logit, dim=0)

        weights = current_weights

    ancestors = D.Categorical(weights[..., -1, 0].permute(1, 0)).sample((pred_batch_size, ))
    current_states = batched_index_select(current_states, 0, ancestors, 1).squeeze(1)
    states, observations = sample_from_prior(model, pred_batch_size, pred_steps, current_states)
    return observations
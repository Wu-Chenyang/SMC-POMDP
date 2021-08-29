import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as D

from typing import NamedTuple
from utils.util import batched_index_select, systematic_sampling
import math


class SMCResult(NamedTuple):
    weights: torch.Tensor
    proposal_log_probs: torch.Tensor
    model_log_probs: torch.Tensor
    log_likelihood: torch.Tensor
    # encoded_obs_log_probs: torch.Tensor

# from memory_profiler import profile

# @profile
identity_twister = lambda x: 1.0
def auxiliary_smc(proposal: nn.Module, model: nn.Module, batched_data: NamedTuple,
            num_particles: int, twister = identity_twister, filtering_objective: bool = False) -> SMCResult:

    # observations = model.encode(batched_data.observations)
    observations = batched_data.observations
    actions = batched_data.actions
    categories = batched_data.prior_mixtures

    batch_size, seq_len, _ = observations.shape

    kwargs = proposal.reset(observations, actions, num_particles)

    current_states, incremental_log_weights, proposal_log_probs, kwargs = proposal.prior_proposal(categories, num_particles, **kwargs)
    prior_distribution = model.prior(categories)
    prior_log_probs = prior_distribution.log_prob(current_states)

    model_log_probs = prior_log_probs[..., None, None]
    proposal_log_probs = proposal_log_probs[..., None, None]
    incremental_log_weights = incremental_log_weights[..., None, None]

    log_particle_num = math.log(num_particles)

    current_logit = model_log_probs - incremental_log_weights
    weights = F.softmax(current_logit, dim=0)
    log_likelihood = (torch.logsumexp(current_logit, dim=0) - log_particle_num).squeeze(-1)

    for i in range(seq_len):
        current_observations = observations[None, :, i, :].expand(num_particles, -1, -1)
        current_actions = actions[None, :, i, :].expand(num_particles, -1, -1)

        # ancestors = residual_sampling(weights[..., -1, 0].permute(1, 0).to("cpu")).to(weights.device)
        ancestors = systematic_sampling(weights[..., -1, 0].permute(1, 0)).permute(1, 0)
        # ancestors = D.Categorical(weights[..., -1, 0].permute(1, 0)).sample((num_particles, ))

        resampled_states = batched_index_select(current_states, 0, ancestors, 1)
        kwargs = batched_index_select(kwargs, 1, ancestors, 2)

        ##### resampled_states.detach()
        current_states, incremental_log_weights, current_proposal_log_probs, kwargs = proposal.transition_proposal(resampled_states, current_actions, current_observations, i, **kwargs)
        incremental_log_weights = incremental_log_weights[..., None, None]
        current_proposal_log_probs = current_proposal_log_probs[..., None, None]

        transition_distribution = model.transition(resampled_states, current_actions)
        current_transition_log_probs = transition_distribution.log_prob(current_states)

        observation_distribution = model.observation(current_actions, current_states)
        current_observation_log_probs = observation_distribution.log_prob(current_observations)

        # with torch.no_grad():
        #     encoded_obs_distribution = model.observation(current_actions, current_states)
        # current_encoded_obs_log_probs = encoded_obs_distribution.log_prob(current_observations)[..., None, None]

        current_model_log_probs = (current_transition_log_probs + current_observation_log_probs)[..., None, None]
        ##### incremental_log_weights.detach()
        current_logit = current_model_log_probs - incremental_log_weights
        current_weights = F.softmax(current_logit, dim=0)
        log_likelihood += (torch.logsumexp(current_logit, dim=0) - log_particle_num).squeeze(-1)

        if filtering_objective:
            weights = torch.cat([weights, current_weights], dim=-2)
        else:
            weights = current_weights
            if i != 0:
                proposal_log_probs = batched_index_select(proposal_log_probs, 0, ancestors, 1)
                model_log_probs = batched_index_select(model_log_probs, 0, ancestors, 1)
                # encoded_probs = batched_index_select(encoded_obs_log_probs, 0, ancestors, 1)

        proposal_log_probs = torch.cat([proposal_log_probs, current_proposal_log_probs], dim=-2)
        model_log_probs = torch.cat([model_log_probs, current_model_log_probs], dim=-2)
        # encoded_obs_log_probs = torch.cat([encoded_obs_log_probs, current_encoded_obs_log_probs], dim=-2)

    return SMCResult(weights=weights,
                     proposal_log_probs=proposal_log_probs,
                     model_log_probs=model_log_probs,
                     log_likelihood=log_likelihood / torch.tensor(seq_len, dtype=torch.float, device=weights.device),
                    #  encoded_obs_log_probs=encoded_obs_log_probs
                     )

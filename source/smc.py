import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import NamedTuple
import math


class SMCResult(NamedTuple):
    weights: torch.Tensor
    proposal_log_probs: torch.Tensor
    model_log_probs: torch.Tensor
    log_likelihood: torch.Tensor
    # encoded_obs_log_probs: torch.Tensor

def systematic_sampling(weights: torch.Tensor) -> torch.Tensor:
    """Sample ancestral index using systematic resampling.
    Get from https://docs.pyro.ai/en/stable/_modules/pyro/infer/smcfilter.html#SMCFilter

    Args:
        log_weight: log of unnormalized weights, tensor
            [batch_shape, num_particles]
    Returns:
        zero-indexed ancestral index: LongTensor [batch_shape, num_particles]
    """
    with torch.no_grad():
        batch_shape, size = weights.shape[:-1], weights.size(-1)
        n = weights.cumsum(-1).mul_(size).add_(torch.rand(batch_shape + (1,), device=weights.device))
        n = n.floor_().long().clamp_(min=0, max=size)
        diff = torch.zeros(batch_shape + (size + 1,), dtype=torch.long, device=weights.device)
        diff.scatter_add_(-1, n, torch.tensor(1, device=weights.device, dtype=torch.long).expand_as(weights))
        ancestors = diff[..., :-1].cumsum(-1).clamp_(min=0, max=size-1)
    return ancestors

def batched_index_select(inputs, dim: int, index: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    if isinstance(inputs, dict):
        return {k: batched_index_select(v, dim, index, batch_dim) for k, v in inputs.items()}
    views = [1 if i != dim else -1 for i in range(len(inputs.shape))]
    views[batch_dim] = inputs.shape[batch_dim]
    expanse = list(inputs.shape)
    expanse[batch_dim] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inputs, dim, index)

# from memory_profiler import profile

# @profile
def smc_pomdp(proposal: nn.Module, model: nn.Module, batched_data: NamedTuple, num_particles: int, filtering_objective: bool = False) -> SMCResult:
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

    batch_size, seq_len, _ = observations.shape

    proposal_args = proposal.reset(observations, actions, num_particles)

    prior_distribution = model.prior(batched_data.prior_mixtures)
    current_states = prior_distribution.rsample((num_particles,))
    prior_log_probs = prior_distribution.log_prob(current_states)

    model_log_probs = prior_log_probs[..., None, None]
    proposal_log_probs = model_log_probs.detach() # model prior serves as the prior at the 1-st timestep
    weights = torch.ones_like(model_log_probs, device=observations.device) / num_particles
    log_likelihood = torch.zeros((batch_size, 1), device=observations.device)
    log_particle_num = math.log(num_particles)
    # encoded_obs_log_probs = torch.ones_like(model_log_probs, device=observations.device)

    for i in range(seq_len):
        current_observations = observations[None, :, i, :].expand(num_particles, -1, -1)
        current_actions = actions[None, :, i, :].expand(num_particles, -1, -1)

        if i == 0:
            resampled_states = current_states
        else:
            # ancestors = residual_sampling(weights[..., -1, 0].permute(1, 0).to("cpu")).to(weights.device)
            ancestors = systematic_sampling(weights[..., -1, 0].permute(1, 0)).permute(1, 0)
            # ancestors = D.Categorical(weights[..., -1, 0].permute(1, 0)).sample((num_particles, ))

            resampled_states = batched_index_select(current_states, 0, ancestors, 1)
            proposal_args = batched_index_select(proposal_args, 1, ancestors, 2)

        current_states, current_proposal_log_probs, proposal_args = proposal.transition_proposal(resampled_states.detach(), current_actions, current_observations, i, **proposal_args)
        current_proposal_log_probs = current_proposal_log_probs[..., None, None]

        transition_distribution = model.transition(resampled_states, current_actions)
        current_transition_log_probs = transition_distribution.log_prob(current_states)

        observation_distribution = model.observation(current_actions, current_states)
        current_observation_log_probs = observation_distribution.log_prob(current_observations)

        # with torch.no_grad():
        #     encoded_obs_distribution = model.observation(current_actions, current_states)
        # current_encoded_obs_log_probs = encoded_obs_distribution.log_prob(current_observations)[..., None, None]

        current_model_log_probs = (current_transition_log_probs + current_observation_log_probs)[..., None, None]
        current_logit = current_model_log_probs - current_proposal_log_probs.detach()
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

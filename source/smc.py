import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from typing import Tuple, NamedTuple


class SMCResult(NamedTuple):
    intermediate_weights: torch.Tensor
    intermediate_trajectories: torch.Tensor
    intermediate_proposal_log_probs: torch.Tensor
    intermediate_model_log_probs: torch.Tensor
    intermediate_encoded_obs_log_probs: torch.Tensor
    final_weights: torch.Tensor
    final_trajectories: torch.Tensor
    final_proposal_log_probs: torch.Tensor
    final_model_log_probs: torch.Tensor
    final_encoded_obs_log_probs: torch.Tensor

def residual_sampling(weights: torch.Tensor) -> torch.Tensor:
    # Runs slowly (possibly an implementation issue)
    num_batch, num_particles = weights.shape

    expected_sampling = num_particles * weights
    deterministic_sampling = torch.floor(expected_sampling).long()
    residual_weights = expected_sampling - deterministic_sampling
    random_samples = num_particles - torch.sum(deterministic_sampling, dim=1)

    ancestors = torch.zeros((num_particles, num_batch), device=weights.device, dtype=torch.int64)
    for i in range(num_batch):
        if random_samples[i].item() < 1.0:
            continue
        residual_distribution = D.Categorical(residual_weights[i])
        random_ancestors = residual_distribution.sample((random_samples[i].item(),))
        deterministic_ancestors = []
        for j in range(num_particles):
            samples = deterministic_sampling[i, j].item()
            if samples > 0:
                deterministic_ancestors += samples * [j]
        ancestors[:, i] = torch.cat((random_ancestors, torch.tensor(deterministic_ancestors, device=weights.device)))
    return ancestors

def systematic_sampling(weights: torch.Tensor) -> torch.Tensor:
    # Get from https://docs.pyro.ai/en/stable/_modules/pyro/infer/smcfilter.html#SMCFilter
    # Something is wrong.
    batch_shape, size = weights.shape[:-1], weights.size(-1)
    n = weights.cumsum(-1).mul_(size).add_(torch.rand(batch_shape + (1,), device=weights.device))
    n = n.floor_().clamp_(min=0, max=size).long()
    diff = weights.new_zeros(batch_shape + (size + 1,))
    diff.scatter_add_(-1, n, torch.ones_like(weights, device=weights.device))
    ancestors = diff[..., :-1].cumsum(-1).swapaxes(0, 1).long()
    return ancestors

def batched_index_select(inputs: torch.Tensor, dim: int, index: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    views = [1 if i != dim else -1 for i in range(len(inputs.shape))]
    views[batch_dim] = inputs.shape[batch_dim]
    expanse = list(inputs.shape)
    expanse[batch_dim] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inputs, dim, index)

def smc_pomdp(proposal: nn.Module, model: nn.Module,
                      example: NamedTuple,
                      num_particles: int) -> SMCResult:
    # Dimensionality Analysis
    #
    # Observations: [batch_size, sequence_length, obs_dim]
    # Weights: [num_particles, batch_size, num_timesteps, 1]
    # Final Weights: [num_particles, batch_size, 1]
    # LSTM State: [num_particles, batch_size, hidden_dim]
    # Intermediate Trajectories: [num_particles, batch_size, num_timesteps, state_dim]
    # Final Trajectories: [num_particles, batch_size, num_timesteps, state_dim]
    # Current States: [num_particles, batch_size, state_dim]
    # Current Observations: [1, batch_size, obs_dim]
    # Proposal Log Probabilities: [num_particles, batch_size, num_timesteps, 1]

    observations = model.encode(example.observations)

    batch_size, seq_len, _ = observations.shape

    lstm_state = proposal.reset(num_particles, batch_size)

    prior_distribution = model.prior(example.prior_mixtures)
    current_states = prior_distribution.sample((num_particles,))
    prior_log_probs = prior_distribution.log_prob(current_states)

    model_log_probs = prior_log_probs[..., None, None]
    final_model_log_probs = model_log_probs

    proposal_log_probs = model_log_probs # model prior serves as the prior at the 1-st timestep
    final_proposal_log_probs = proposal_log_probs

    encoded_obs_log_probs = torch.ones_like(model_log_probs, device=observations.device)
    final_encoded_obs_log_probs = encoded_obs_log_probs

    weights = torch.ones_like(model_log_probs, device=observations.device) / num_particles

    intermediate_trajectories = current_states[..., None, :]
    final_trajectories = current_states[..., None, :]

    for i in range(seq_len):
        current_observations = observations[None, :, i, :].expand(num_particles, -1, -1)
        current_actions = example.actions[None, :, i, :].expand(num_particles, -1, -1)
        current_pred_targets = example.pred_targets[None, :, i, :].expand(num_particles, -1, -1)

        # ancestors = residual_sampling(weights[..., i - 1, 0].permute(1, 0).to("cpu")).to(weights.device)
        # ancestors = systematic_sampling(weights[..., i - 1, 0].permute(1, 0))
        ancestors = D.Categorical(weights[..., i - 1, 0].permute(1, 0)).sample((num_particles, ))

        resampled_trajectories = batched_index_select(final_trajectories, 0, ancestors, 1)
        resampled_proposal_log_probs = batched_index_select(final_proposal_log_probs, 0, ancestors, 1)
        resampled_model_log_probs = batched_index_select(final_model_log_probs, 0, ancestors, 1)
        resampled_encoded_probs = batched_index_select(final_encoded_obs_log_probs, 0, ancestors, 1)
        resampled_lstm_h = batched_index_select(lstm_state[0], 1, ancestors, 2)
        resampled_lstm_c = batched_index_select(lstm_state[1], 1, ancestors, 2)
        resampled_lstm_state = (resampled_lstm_h, resampled_lstm_c)

        previous_states = resampled_trajectories[..., i, :]

        proposal_distribution, lstm_state = proposal.transition_proposal(previous_states, current_actions, current_observations, resampled_lstm_state)
        current_states = proposal_distribution.sample()
        current_proposal_log_probs = proposal_distribution.log_prob(current_states)[..., None, None]

        transition_distribution = model.transition(previous_states, current_actions)
        current_transition_log_probs = transition_distribution.log_prob(current_states)

        observation_distribution = model.observation(current_actions, current_states)
        current_observation_log_probs = observation_distribution.log_prob(current_observations)

        prediction_distribution = model.predict(current_states)
        current_prediction_log_probs = prediction_distribution.log_prob(current_pred_targets)

        with torch.no_grad():
            encoded_obs_distribution = model.observation(current_actions, current_states)
        current_encoded_obs_log_probs = encoded_obs_distribution.log_prob(current_observations)[..., None, None]

        current_model_log_probs = (current_transition_log_probs + current_observation_log_probs + current_prediction_log_probs)[..., None, None]
        current_weights = F.softmax(current_model_log_probs - current_proposal_log_probs, dim=0)
        weights = torch.cat([weights, current_weights], dim=-2)

        current_trajectories = current_states[..., None, :]
        intermediate_trajectories = torch.cat([intermediate_trajectories, current_trajectories], dim=-2)
        final_trajectories = torch.cat([resampled_trajectories, current_trajectories], dim=-2)

        proposal_log_probs = torch.cat([proposal_log_probs, current_proposal_log_probs], dim=-2)
        final_proposal_log_probs = torch.cat([resampled_proposal_log_probs, current_proposal_log_probs], dim=-2)

        model_log_probs = torch.cat([model_log_probs, current_model_log_probs], dim=-2)
        final_model_log_probs = torch.cat([resampled_model_log_probs, current_model_log_probs], dim=-2)

        encoded_obs_log_probs = torch.cat([encoded_obs_log_probs, current_encoded_obs_log_probs], dim=-2)
        final_encoded_obs_log_probs = torch.cat([resampled_encoded_probs, current_encoded_obs_log_probs], dim=-2)

    return SMCResult(intermediate_weights=weights,
                     intermediate_trajectories=intermediate_trajectories,
                     intermediate_proposal_log_probs=proposal_log_probs,
                     intermediate_model_log_probs=model_log_probs,
                     intermediate_encoded_obs_log_probs=encoded_obs_log_probs,
                     final_weights=weights[..., -1, :],
                     final_trajectories=final_trajectories,
                     final_proposal_log_probs=final_proposal_log_probs,
                     final_model_log_probs=final_model_log_probs,
                     final_encoded_obs_log_probs=final_encoded_obs_log_probs)

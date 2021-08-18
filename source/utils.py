import torch
import torch.nn as nn
from smc import batched_index_select
from typing import NamedTuple

def global_grad_norm(params):
    grad_norm = 0.0
    for param in params:
        if param.grad == None:
            continue
        grad_norm = max(grad_norm, torch.max(torch.abs(param.grad.data)).item())
    return grad_norm

def sample_from_prior(model: nn.Module, batch_size: int, seq_len: int, start_states = None, future_actions = None) -> torch.Tensor:
# sample from model prior
# Input: current_states None
#                       BatchDim * StateDim
#                       StateDim
#        future_actions None
#                       Seqlen * BatchDim * ActionDim
#                       BatchDim * ActionDim 
#                       ActionDim
    observations = None
    device = next(model.parameters()).device
    if start_states == None:
        prior_distribution = model.prior(torch.randint(4, (batch_size, 1), device=device))
        current_states = prior_distribution.sample()
    elif len(start_states.shape) == 1:
        current_states = start_states.expand(batch_size, -1)
    else:
        assert start_states.shape[0] == batch_size
        current_states = start_states
    states = current_states.unsqueeze(0)

    if future_actions == None:
        future_actions = torch.zeros(seq_len, batch_size, 2, device=device)
    elif len(future_actions.shape) != 3:
        future_actions = future_actions.expand(seq_len, batch_size, future_actions.shape[-1])
    else:
        assert future_actions.shape[1] == batch_size

    for i in range(seq_len):
        current_actions = future_actions[i]
        current_states = model.transition(current_states, current_actions).sample()
        current_observations = model.observation(current_actions, current_states).sample()

        states = torch.cat((states, current_states.unsqueeze(0)), dim=0)
        if i == 0:
            observations = current_observations.unsqueeze(0)
        else:
            observations = torch.cat((observations, current_observations.unsqueeze(0)), dim=0)

    return states, observations

def self_contrastive_loss(states: torch.Tensor, actions: torch.Tensor, observations: torch.Tensor, proposal: nn.Module):
    proposal.reset(observations, actions)
    batch_size = observations.shape[1]
    h = proposal.h0.expand(-1, batch_size, -1).contiguous()
    c = proposal.c0.expand(-1, batch_size, -1).contiguous()
    hidden_futures, _ = proposal.lstm(torch.flip(torch.cat((observations, actions), dim=-1), (0,)), (h, c))
    future_encodings = proposal.future_encoder(torch.flip(hidden_futures, (0,)))
    state_encodings = proposal.state_encoder(states)
    D = torch.sigmoid(torch.matmul(state_encodings, future_encodings.transpose(-1, -2)))

    I = torch.eye(batch_size, device=states.device)
    mul = 2.0 * I - 1.0
    bias = 1.0 - I + 1e-7
    factor = (batch_size - 2.0 / (batch_size - 1.0)) * I + 1.0 / (batch_size - 1.0)

    return -torch.mean(factor * (D * mul + bias).log()) * batch_size
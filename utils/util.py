import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import (
    Independent,
    MultivariateNormal,
    Normal,
    register_kl
)
import math

eps = 1e-5
max_deviation = 10.0

def softclamp(input: torch.Tensor, min: float, max: float, boundary = None) -> torch.Tensor:
    assert max > min
    if boundary is not None:
        assert 0.0 < boundary <= (max - min) / 2.0
    else:
        boundary = 1e-5
    logsigmoid_offset = max + math.log(1.0 - math.exp(-boundary))
    softplus_offset = min - math.log(1.0 - math.exp(-boundary))
    output = torch.zeros_like(input, device=input.device)
    large_index = input > max - boundary
    small_index = input < min + boundary
    med_index = torch.logical_not(torch.logical_or(large_index, small_index))
    output[large_index] = max + F.logsigmoid(input[large_index] - logsigmoid_offset)
    output[small_index] = min + F.softplus(input[small_index] - softplus_offset)
    output[med_index] = input[med_index]
    return output

class Softclamp(nn.Module):
    def __init__(self, min: float, max: float, boundary: float = 1e-5) -> None:
        super(Softclamp, self).__init__()
        self.min = min
        self.max = max
        self.boundary = boundary

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return softclamp(input, self.min, self.max, self.boundary)

    def extra_repr(self) -> str:
        return 'min={}, max={}, boundary={}'.format(self.min, self.max, self.boundary)

def global_grad_norm(params):
    grad_norm = 0.0
    for param in params:
        if param.grad == None:
            continue
        grad_norm = max(grad_norm, torch.max(torch.abs(param.grad.data)).item())
    return grad_norm

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

def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


# this function takes the hidden state as output by the PyTorch rnn and
# unpacks it it; it also reverses each sequence temporally
def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output
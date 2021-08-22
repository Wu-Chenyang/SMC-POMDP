import torch
import torch.nn as nn
from torch.distributions import (
    Independent,
    MultivariateNormal,
    Normal,
    register_kl
)
import math

eps = 1e-5
max_deviation = 10.0

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

@register_kl(Independent, MultivariateNormal)
def _kl_independent_mvn(p, q):
    if isinstance(p.base_dist, Normal) and p.reinterpreted_batch_ndims == 1:
        dim = q.event_shape[0]
        p_cov = p.base_dist.scale ** 2
        q_precision = q.precision_matrix.diagonal(dim1=-2, dim2=-1)
        return (
            0.5 * (p_cov * q_precision).sum(-1)
            - 0.5 * dim * (1 + math.log(2 * math.pi))
            - q.log_prob(p.base_dist.loc)
            - p.base_dist.scale.log().sum(-1)
        )

    raise NotImplementedError

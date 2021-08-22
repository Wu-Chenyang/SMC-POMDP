import torch
import torch.nn as nn
from utils import eps

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
    bias = 1.0 - I + eps
    factor = (batch_size - 2.0 / (batch_size - 1.0)) * I + 1.0 / (batch_size - 1.0)

    return -torch.mean(factor * (D * mul + bias).log()) * batch_size
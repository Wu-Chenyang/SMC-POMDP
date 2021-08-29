import torch.nn as nn
import torch

from utils.util import eps

class Twister(nn.Module):
    def prior_proposal(self, num_particles: int, **kwargs: dict):
        dists, mean_square_scale = self.prior_proposal_nn(self.hidden_futures[-1])
        states = dists.sample((num_particles,))
        state_encoding = self.state_encoder(states)
        future_encoding = self.future_encodings[-1]
        D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        proposal_log_probs = dists.log_prob(states)
        incremental_log_weights = proposal_log_probs - future_log_likelihood
        return states, incremental_log_weights, proposal_log_probs, {}

    def transition_proposal(self, previous_state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, time_step: int, **kwargs: dict):
        # time_step start from 1
        future = self.hidden_futures[-time_step-1, :, :].unsqueeze(0)
        dists = self.proposal_nn(torch.cat((previous_state, future.expand(previous_state.shape[0], -1, -1)), axis=-1))
        next_states = dists.sample()
        # with torch.no_grad():
        state_encoding = self.state_encoder(previous_state)
        future_encoding = self.future_encodings[-time_step-1]
        D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
        if time_step + 2 <= self.hidden_futures.shape[2]:
            next_state_encoding = self.state_encoder(next_states)
            next_future_encoding = self.future_encodings[-time_step-2]
            D_next = torch.sigmoid((next_state_encoding * next_future_encoding).sum(-1))
        else:
            D_next = torch.full(D.shape, 0.5, device=previous_state.device)
        #############
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        next_future_log_likelihood = torch.log(D_next / (1 - D_next + eps) + eps)
        proposal_log_probs = dists.log_prob(next_states)
        incremental_log_weights = future_log_likelihood - next_future_log_likelihood + proposal_log_probs
        return next_states, incremental_log_weights, proposal_log_probs, {}

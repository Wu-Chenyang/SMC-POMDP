import torch
import torch.nn.functional as F

from utils.util import eps
from  model.proposals import TASMCProposal
from model.building_blocks import CGMM, CGM

class BatteryTASMC(TASMCProposal):
    def __init__(self, state_dim: int = 5, action_dim: int = 3, obs_dim: int = 5, mixture_num: int = 3, hidden_dim: int = 50, rnn_num: int = 2, num_hidden_layers: int = 2, category_num: int = 4):
        super().__init__(self, state_dim, action_dim, obs_dim,
                        mixture_num, hidden_dim,
                        rnn_num, num_hidden_layers)
        self.category_num = category_num
        if mixture_num > 1:
            self.prior_proposal_nn = CGMM(category_num + hidden_dim, state_dim, hidden_dim, num_hidden_layers, mixture_num)
        else:
            self.prior_proposal_nn = CGM(category_num + hidden_dim, state_dim, hidden_dim, num_hidden_layers)

    def prior_proposal(self, category: torch.Tensor, batch_shape, **kwargs: dict):
        category = F.one_hot(category.squeeze(-1), num_classes=self.category_num)
        future = self.hidden_futures[-1, :, :]
        dists, mean_square_scale = self.prior_proposal_nn(torch.cat((future, category), dim=-1))
        states = dists.sample((num_particles,))
        state_encoding = self.state_encoder(states)
        future_encoding = self.future_encodings[-1]
        D = torch.sigmoid((state_encoding * future_encoding).sum(-1))
        future_log_likelihood = torch.log(D / (1 - D + eps) + eps)
        proposal_log_probs = dists.log_prob(states)
        incremental_log_weights = proposal_log_probs - future_log_likelihood
        return states, incremental_log_weights, proposal_log_probs, {}
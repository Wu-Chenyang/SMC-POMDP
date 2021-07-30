import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch import autograd

import time
import os
import fire
import random

import numpy as np

from utils import global_grad_norm
from proposal_models import NASMCProposal
from datasets import BatteryDataset
from smc import smc_pomdp
from models import BatteryModel

# import multiprocessing


class NASMCTrainer:
    def run(self,
            run_dir: str = './runs/',
            proposal_lr: float = 1e-4,
            model_lr: float = 1e-4,
            state_dim: int = 5,
            action_dim: int = 3,
            obs_dim: int = 5,
            num_steps: int = 1,
            save_decimation: int = 100,
            num_particles: int = 1000,
            sequence_length: int = 4096,
            batch_size: int = 1,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            data_dir: str = "data",
            seed: int = 95):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')

        cell_list = []
        cell_list += [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, item))]

        device = torch.device(device_name)

        proposal = NASMCProposal(state_dim, action_dim, obs_dim)
        model = BatteryModel(
            state_dim = state_dim, action_dim = action_dim,
            trans_mixture_num = 2, trans_hidden_dim = 50,
            obs_mixture_num = 2, obs_hidden_dim = 50,
            pred_mixture_num = 2, pred_hidden_dim = 10, device=None,
            # Encoding Net
            obs_channel = 4, sequence_length = sequence_length, channels = [8] * 5,
            kernel_sizes = [4] * 6, strides = [4] * 6
        )

        proposal_optimizer = torch.optim.Adam(proposal.parameters(), lr=proposal_lr)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)

        proposal.to(device)
        model.to(device)

        log_dir = None
        step = 1
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            proposal_optimizer.load_state_dict(checkpoint['proposal_optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            sequence_length = checkpoint['sequence_length']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        proposal.train()
        model.train()

        # multiprocessing.set_start_method('spawn')
        assert batch_size == 1
        dl = DataLoader(BatteryDataset(cell_list), batch_size=batch_size, shuffle=True, num_workers=2)
        start_time = time.time()
        for i, cell in zip(range(num_steps), dl):
            cell = cell.to(device)

            smc_result = smc_pomdp(proposal, model, cell, num_particles)

            ### Proposal Training

            # proposal_loss = -torch.sum(
            #     smc_result.intermediate_weights.detach() *
            #     smc_result.intermediate_proposal_log_probs) / batch_size
            proposal_loss = -torch.sum(
                smc_result.final_weights.detach() *
                smc_result.final_proposal_log_probs.squeeze(-1)) / batch_size

            proposal_optimizer.zero_grad()
            proposal_loss.backward(retain_graph=True)
            proposal_optimizer.step()

            ### Model Training

            # model_loss = -torch.sum(
            #     smc_result.intermediate_weights.detach() *
            #     smc_result.intermediate_model_log_probs) / batch_size
            # encoder_loss = -torch.sum(smc_result.imtermediate_weights.detach() *
            #     smc_result.imtermediate_encoded_obs_log_probs.squeeze(-1)) / batch_size

            model_loss = -torch.sum(
                smc_result.final_weights.detach() *
                smc_result.final_model_log_probs.squeeze(-1)) / batch_size
            encoder_loss = torch.sum(smc_result.final_weights.detach() *
                smc_result.final_encoded_obs_log_probs.squeeze(-1)) / batch_size

            model_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            encoder_loss.backward()
            model_optimizer.step()

            # Recording

            proposal_grad_norm = global_grad_norm(proposal.parameters())
            model_grad_norm = global_grad_norm(model.parameters())

            summary_writer.add_scalar('proposal_loss/train', proposal_loss, step)
            summary_writer.add_scalar('proposal_gradient', proposal_grad_norm, step)
            summary_writer.add_scalar('model_loss/train', model_loss, step)
            summary_writer.add_scalar('model_gradient', model_grad_norm, step)

            print(f'time = {time.time()-start_time:.1f} step = {step}  proposal_loss = {proposal_loss.item():.1f}  proposal_gradient = {proposal_grad_norm:.1f}  model_loss = {model_loss.item():.1f}  model_gradient = {model_grad_norm:.1f}')

            step += 1
            if step % save_decimation == 0:
                torch.save(
                    dict(proposal=proposal.state_dict(),
                         model=model.state_dict(),
                         proposal_optimizer=proposal_optimizer.state_dict(),
                         step=step,
                         num_particles=num_particles,
                         sequence_length=sequence_length,
                         log_dir=summary_writer.log_dir), checkpoint_path)

        summary_writer.flush()

        torch.save(
            dict(proposal=proposal.state_dict(),
                 model=model.state_dict(),
                 proposal_optimizer=proposal_optimizer.state_dict(),
                 step=step,
                 num_particles=num_particles,
                 sequence_length=sequence_length,
                 log_dir=summary_writer.log_dir), checkpoint_path)


if __name__ == '__main__':
    fire.Fire(NASMCTrainer)

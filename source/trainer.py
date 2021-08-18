import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch import autograd

import time
import os
import fire
import random

import numpy as np
import matplotlib.pyplot as plt

from utils import global_grad_norm, sample_from_prior, self_contrastive_loss
from proposal_models import NASMCProposal, TASMCProposal
# from datasets import BatteryDataset
from processed_datasets import BatteryDataset
from smc import smc_pomdp
from models import BatteryModel

from predicting import smc_prediction

# import multiprocessing


class NASMCTrainer:
    def run(self,
            run_dir: str = './runs/',
            proposal_lr: float = 1e-3,
            discriminator_lr: float = 1e-2,
            model_lr: float = 1e-4,
            state_dim: int = 5,
            action_dim: int = 2,
            obs_dim: int = 13,
            training_epochs: int = 100000,
            save_interval: int = 10,
            test_interval: int = 10,
            num_particles: int = 1000,
            sequence_length: int = 50,
            batch_size: int = 32,
            filtering_objective: bool = False,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            data_dir: str = "processed_data",
            seed: int = 100):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')

        cell_list = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, item))]

        device = torch.device(device_name)

        proposal = TASMCProposal(state_dim = state_dim, action_dim = action_dim, obs_dim = obs_dim,
            mixture_num = 3, hidden_dim = 50, lstm_num = 2, num_hidden_layers = 2)
        model = BatteryModel(
            state_dim = state_dim, action_dim = action_dim, obs_dim = obs_dim,
            trans_mixture_num = 1, trans_hidden_dim = 20, trans_num_hidden_layers = 1,
            obs_mixture_num = 1, obs_hidden_dim = 20, obs_num_hidden_layers = 1,
            device = None
            # # Encoding Net
            # obs_channel = 4, sequence_length = 4096, channels = [8] * 6,
            # kernel_sizes = [4] * 6, strides = [4] * 6
        )

        proposal.to(device)
        model.to(device)

        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': model_lr},
                                    {'params': proposal.proposal_parameters(), 'lr': proposal_lr},
                                    {'params': proposal.discriminator_parameters(), 'lr': discriminator_lr}])

        log_dir = None
        step = 1
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        # multiprocessing.set_start_method('spawn')
        start_time = time.time()
        for i in range(training_epochs):
            for batched_data in DataLoader(BatteryDataset(cell_list, sequence_length), batch_size=batch_size, shuffle=True,
                num_workers=3
            ):
                optimizer.zero_grad()
                proposal.train()
                model.train()

                batched_data = batched_data.to(device)

                smc_result = smc_pomdp(proposal, model, batched_data, num_particles, filtering_objective)

                ### Proposal Training

                proposal_loss = -torch.mean(
                    torch.sum(
                    smc_result.weights.detach() *
                    smc_result.proposal_log_probs, dim=0))

                proposal_loss.backward()

                ### Model Training

                # Variational Objectives
                # model_loss = -torch.mean(smc_result.log_likelihood)

                # Fisher's identity
                model_loss = -torch.mean(
                    torch.sum(
                    smc_result.weights.detach() *
                    smc_result.model_log_probs, dim=0))

                # encoder_loss = torch.mean(
                #     torch.sum(
                #     smc_result.weights.detach() *
                #     smc_result.encoded_obs_log_probs, dim=0))
                
                model_loss.backward()

                # encoder_loss.backward()

                # torch.autograd.set_detect_anomaly(True)
                # for name, param in model.named_parameters():
                #     print(name, torch.isfinite(param.grad).all())

                # Discriminator Learning
                actions = torch.zeros(sequence_length, num_particles, 2, device=device)
                states, observations = sample_from_prior(model, num_particles, sequence_length, future_actions=actions)
                discriminator_loss = self_contrastive_loss(states[:-1], actions, observations, proposal)
                discriminator_loss.backward()

                optimizer.step()

                # Recording

                proposal_grad_norm = global_grad_norm(proposal.proposal_parameters())
                discriminator_grad_norm = global_grad_norm(proposal.discriminator_parameters())
                model_grad_norm = global_grad_norm(model.parameters())

                proposal_loss = proposal_loss.item()
                model_loss = model_loss.item()
                discriminator_loss = discriminator_loss.item()
                # encoder_loss = model_loss + encoder_loss.item()

                summary_writer.add_scalar('proposal_loss/train', proposal_loss, step)
                summary_writer.add_scalar('proposal_gradient', proposal_grad_norm, step)
                summary_writer.add_scalar('discriminator_loss/train', discriminator_loss, step)
                summary_writer.add_scalar('discriminator_gradient', discriminator_grad_norm, step)
                summary_writer.add_scalar('model_loss/train', model_loss, step)
                # summary_writer.add_scalar('encoder_loss/train', encoder_loss, step)
                summary_writer.add_scalar('model_gradient', model_grad_norm, step)

                # Outputting
                print(f'time = {time.time()-start_time:.1f} step = {step} ' +
                        f'proposal_loss = {proposal_loss:.1f} proposal_gradient = {proposal_grad_norm:.2f} ' +
                        f'discriminator_loss = {discriminator_loss:.1f} discriminator_gradient = {discriminator_grad_norm:.2f} ' +
                        f'model_loss = {model_loss:.1f}  model_gradient = {model_grad_norm:.2f}'
                        )

                # Writing summary
                step += 1
                if step % save_interval == 0:
                    torch.save(
                        dict(proposal=proposal.state_dict(),
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            step=step,
                            num_particles=num_particles,
                            log_dir=summary_writer.log_dir), checkpoint_path)
                if step % test_interval == 0:
                    proposal.eval()
                    model.eval()
                    data = next(iter(DataLoader(BatteryDataset(cell_list, sequence_length), batch_size=1, shuffle=True))).to(device)
                    assert sequence_length >= 500
                    future_observations = smc_prediction(proposal, model, data.sub_sequence(500), 1000, 1500, 10)
                    future_capacity = future_observations[:, :, -1].cpu()
                    plt.plot(data.observations[0, :, -1].cpu())
                    for i in range(future_capacity.shape[1]):
                        plt.plot(range(500, 2000), future_capacity[:, i])
                    summary_writer.add_figure('filtering/test', plt.gcf(), step)
                    plt.close()

                summary_writer.flush()

            torch.save(
                dict(proposal=proposal.state_dict(),
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    num_particles=num_particles,
                    log_dir=summary_writer.log_dir), checkpoint_path)


if __name__ == '__main__':
    fire.Fire(NASMCTrainer)

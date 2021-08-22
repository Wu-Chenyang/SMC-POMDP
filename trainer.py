import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch import autograd

import time
import os
import fire
import random

import numpy as np
import matplotlib.pyplot as plt

from utils.util import global_grad_norm, sample_from_prior
from proposal_models import NASMCProposal, TASMCProposal
# from datasets import BatteryDataset
from processed_datasets import BatteryDataset
from loss.smc import smc_pomdp
from loss.self_contrastive import self_contrastive_loss
from models import BatteryModel
from itertools import chain
import pickle as pkl

from predicting import smc_prediction

# import multiprocessing
import re
checkpoint_ptn = re.compile('(\d+)\.pt\Z')

def get_cell_list(data_dir, seq_len):
    cell_list = []
    for item in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, item)):
            cell = os.path.join(data_dir, item)
            with open(cell, 'rb') as f:
                data = pkl.load(f)
                if len(data['state_information']) >= seq_len:
                    cell_list.append(cell)
    return cell_list


class Trainer:
    def run(self,
            run_dir: str = './runs/',
            checkpoint_dir: str = './runs/',
            checkpoint_path = None,
            proposal_lr: float = 1e-3,
            discriminator_lr: float = 1e-2,
            model_lr: float = 1e-4,
            update_lr: bool = False,
            state_dim: int = 10,
            action_dim: int = 2,
            obs_dim: int = 13,
            training_epochs: int = 40,
            save_interval: int = 10,
            test_interval: int = 10,
            num_particles: int = 1000,
            sequence_length = 50,
            batch_size: int = 32,
            filtering_objective: bool = False,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            data_dir: str = "processed_data",
            seed = None,
            verbose: bool = False,
            debug: bool = False):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.autograd.set_detect_anomaly(debug)

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if checkpoint_path == None:
            checkpoint_path = [item for item in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, item)) and checkpoint_ptn.search(item) != None]
            if len(checkpoint_path) > 0:
                checkpoint_path = os.path.join(checkpoint_dir, str(max(int(checkpoint_ptn.search(item).group(1)) for item in checkpoint_path)) + '.pt')
            else:
                checkpoint_path = None
        elif not os.path.isfile(checkpoint_path):
            raise ValueError('checkpoint_path should point to a checkpoint file or set to None')

        device = torch.device(device_name)

        proposal = TASMCProposal(state_dim = state_dim, action_dim = action_dim, obs_dim = obs_dim,
            mixture_num = 3, hidden_dim = 50, rnn_num = 2, num_hidden_layers = 2)
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

        steps_per_epoch = 0
        data_loaders = []
        if isinstance(sequence_length, int):
            cell_list = get_cell_list(data_dir, sequence_length)
            data_loader = DataLoader(BatteryDataset(cell_list, sequence_length), batch_size=batch_size, shuffle=True, num_workers=2)
            steps_per_epoch += len(data_loader)
            data_loaders.append(data_loader)
        else:
            for seq_len in sequence_length:
                cell_list = get_cell_list(data_dir, seq_len)
                data_loader = DataLoader(BatteryDataset(cell_list, seq_len), batch_size=batch_size, shuffle=True, num_workers=2)
                steps_per_epoch += len(data_loader)
                data_loaders.append(data_loader)

        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': model_lr},
                                    {'params': proposal.proposal_parameters(), 'lr': proposal_lr},
                                    {'params': proposal.discriminator_parameters(), 'lr': discriminator_lr}])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[model_lr, proposal_lr, discriminator_lr],
                                                        steps_per_epoch=steps_per_epoch,
                                                        epochs=training_epochs, verbose=verbose)

        log_dir = None
        step = 1
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not update_lr:
                scheduler.load_state_dict(checkpoint['scheduler'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        # multiprocessing.set_start_method('spawn')
        start_time = time.time()

        for epoch_idx in range(training_epochs):
            for batch_idx, batched_data in enumerate(chain(*data_loaders)):
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

                # Discriminator Learning
                actions = torch.zeros(2000, num_particles, 2, device=device)
                states, observations = sample_from_prior(model, num_particles, 2000, future_actions=actions)
                discriminator_loss = self_contrastive_loss(states[:-1], actions, observations, proposal)
                discriminator_loss.backward()

                # clip_grad_norm_(chain(model.parameters(), proposal.parameters()), 10, 2, True)
                optimizer.step()
                scheduler.step()

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
                            scheduler=scheduler.state_dict(),
                            step=step,
                            num_particles=num_particles,
                            log_dir=summary_writer.log_dir), os.path.join(checkpoint_dir, str(step) + '.pt'))
                if step % test_interval == 0:
                    proposal.eval()
                    model.eval()
                    data = next(iter(DataLoader(BatteryDataset(get_cell_list(data_dir, 2000), 2000), batch_size=1, shuffle=True))).to(device)
                    future_observations = smc_prediction(proposal, model, data.sub_sequence(500), 1000, 1500, 2)
                    future_capacity = future_observations[:, :, -1].cpu()
                    for epoch_idx in range(future_capacity.shape[1]):
                        plt.plot(range(500, 2000), future_capacity[:, epoch_idx])
                    plt.plot(data.observations[0, :, -1].cpu())
                    summary_writer.add_figure('filtering/test', plt.gcf(), step)
                    plt.close()

                summary_writer.flush()

        torch.save(
            dict(proposal=proposal.state_dict(),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=step,
                num_particles=num_particles,
                log_dir=summary_writer.log_dir), os.path.join(checkpoint_dir, str(step) + '.pt'))


if __name__ == '__main__':
    fire.Fire(Trainer)

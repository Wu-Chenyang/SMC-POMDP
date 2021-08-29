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
from model.guides import SmoothingProposal
from model.models import DMM
from loss.auxiliary_smc import auxiliary_smc
from loss.self_contrastive import self_contrastive_loss
from itertools import chain

from predicting import smc_prediction

from battery.datasets import BatteryDataset, get_cell_list

import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceMeanField_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)

# import multiprocessing
import re
checkpoint_ptn = re.compile('(\d+)\.pt\Z')

class Trainer:
    def run(self,
            run_dir: str = './runs/',
            checkpoint_dir: str = './runs/',
            checkpoint_path = None,
            update_lr: bool = False,

            training_epochs: int = 40,
            save_interval: int = 10,
            test_interval: int = 10,
            num_particles: int = 1000,
            sequence_length = 50,
            batch_size: int = 32,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",

            dataset: str = "battery",
            config: dict = {'state_dim': 3, 'static_state_dim': 2},
            model: str = "DMM",
            model_lr: float = 1e-3,
            model_config: dict = {},
            guide: str = "SmoothingProposal",
            guide_lr: float = 1e-3,
            guide_config: dict = {'proposal_mixture_num': 1, 'rnn_num': 1},
            loss: str = "Trace_ELBO",
            loss_config: dict = {'num_particles': 1, 'vectorize_particles': True},
            optimizer: str = "ClippedAdam",
            optimizer_config: dict = {'betas': (0.95, 0.999), 'clip_norm': 10.0, 'lrd': 0.99999, 'weight_decay': 1.0},
            inference_algorithm: str = "SVI",
            infer_config: dict = {},
            seed = None,
            verbose: bool = False,
            debug: bool = False):

        # Set seeds
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Debug
        torch.autograd.set_detect_anomaly(debug)
        
        # Load Dataset
        if dataset is "battery":
            data_dir = 'battery/data'
            cell_list = get_cell_list(data_dir, 1)
            data_loader = DataLoader(BatteryDataset(cell_list, sequence_length), batch_size=batch_size, shuffle=True, num_workers=2)
            steps_per_epoch = len(data_loader)
            action_dim = 1
            obs_dim = 11
            static_info_dim = 4
            with_initial_obs=False
        else:
            raise ValueError(f'{dataset} is not a valid dataset')

        # Load the checkpoint specified by checkpoint_path or the latest checkpoint in checkpoint_dir
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if checkpoint_path is None:
            checkpoint_path = [item for item in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, item)) and checkpoint_ptn.search(item) != None]
            if len(checkpoint_path) > 0:
                checkpoint_path = os.path.join(checkpoint_dir, str(max(int(checkpoint_ptn.search(item).group(1)) for item in checkpoint_path)) + '.pt')
            else:
                checkpoint_path = None
        elif not os.path.isfile(checkpoint_path):
            raise ValueError('checkpoint_path should point to a checkpoint file or set to None')

        # Set training device
        device = torch.device(device_name)

        # Make sure hidden state dimensions are specified
        assert config.get('state_dim', None) is not None
        assert config.get('static_state_dim', None) is not None

        # Initialize Model
        if model == "DMM":
            model = DMM(action_dim=action_dim, obs_dim=obs_dim,
                with_initial_obs=with_initial_obs, static_info_dim=static_info_dim,
                **config, **model_config
            )
        else:
            raise NotImplementedError(f'{model} is not implemented.')
        model.to(device)

        # Initialize Guide
        if guide == "SmoothingProposal":
            guide = SmoothingProposal(action_dim=action_dim, obs_dim=obs_dim,
                with_initial_obs=with_initial_obs, static_info_dim=static_info_dim,
                **config, **guide_config
            )
        else:
            raise NotImplementedError(f'{guide} is not implemented.')
        guide.to(device)

        if optimizer == 'ClippedAdam':
            optimizer = pyro.optim.ClippedAdam([{'params': model.parameters(), 'lr': model_lr},
                                        {'params': guide.parameters(), 'lr': guide_lr}], **optimizer_config)
        else:
            raise NotImplementedError(f'{optimizer} is not supported yet')

        # optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': model_lr},
        #                             {'params': guide.parameters(), 'lr': guide_lr},])
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[model_lr, guide_lr],
        #                                                 steps_per_epoch=steps_per_epoch,
        #                                                 epochs=training_epochs, verbose=verbose)

        if loss == 'Trace_ELBO':
            loss = Trace_ELBO(**loss_config)
        elif loss == 'TraceMeanField_ELBO':
            loss = TraceMeanField_ELBO(**loss_config)
        else:
            raise NotImplementedError(f'{loss} is not implemented.')

        if inference_algorithm == 'SVI':
            inference_algorithm = SVI(model.model, guide.guide, optimizer, loss)
        else:
            raise NotImplementedError(f'{inference_algorithm} is not implemented.')

        # Save model
        log_dir = None
        step = 1
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            guide.load_state_dict(checkpoint['guide'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            num_particles = checkpoint['num_particles']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        # multiprocessing.set_start_method('spawn')
        start_time = time.time()

        for epoch_idx in range(training_epochs):
            for batch_idx, batched_data in enumerate(chain(*data_loaders)):
                optimizer.zero_grad()
                guide.train()
                model.train()

                batched_data = batched_data.to(device)

                smc_result = smc_pomdp(guide, model, batched_data, num_particles, filtering_objective)

                ### Proposal Training

                guide_loss = -torch.mean(
                    torch.sum(
                    smc_result.weights.detach() *
                    smc_result.guide_log_probs, dim=0))

                guide_loss.backward()

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
                discriminator_loss = self_contrastive_loss(states[:-1], actions, observations, guide)
                discriminator_loss.backward()

                # clip_grad_norm_(chain(model.parameters(), guide.parameters()), 10, 2, True)
                optimizer.step()
                scheduler.step()

                # Recording

                guide_grad_norm = global_grad_norm(guide.guide_parameters())
                discriminator_grad_norm = global_grad_norm(guide.discriminator_parameters())
                model_grad_norm = global_grad_norm(model.parameters())

                guide_loss = guide_loss.item()
                model_loss = model_loss.item()
                discriminator_loss = discriminator_loss.item()
                # encoder_loss = model_loss + encoder_loss.item()

                summary_writer.add_scalar('guide_loss/train', guide_loss, step)
                summary_writer.add_scalar('guide_gradient', guide_grad_norm, step)
                summary_writer.add_scalar('discriminator_loss/train', discriminator_loss, step)
                summary_writer.add_scalar('discriminator_gradient', discriminator_grad_norm, step)
                summary_writer.add_scalar('model_loss/train', model_loss, step)
                # summary_writer.add_scalar('encoder_loss/train', encoder_loss, step)
                summary_writer.add_scalar('model_gradient', model_grad_norm, step)

                # Outputting
                print(f'time = {time.time()-start_time:.1f} step = {step} ' +
                        f'guide_loss = {guide_loss:.1f} guide_gradient = {guide_grad_norm:.2f} ' +
                        f'discriminator_loss = {discriminator_loss:.1f} discriminator_gradient = {discriminator_grad_norm:.2f} ' +
                        f'model_loss = {model_loss:.1f}  model_gradient = {model_grad_norm:.2f}'
                        )

                # Writing summary
                step += 1
                if step % save_interval == 0:
                    torch.save(
                        dict(guide=guide.state_dict(),
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            step=step,
                            num_particles=num_particles,
                            log_dir=summary_writer.log_dir), os.path.join(checkpoint_dir, str(step) + '.pt'))
                if step % test_interval == 0:
                    guide.eval()
                    model.eval()
                    data = next(iter(DataLoader(BatteryDataset(get_cell_list(data_dir, 2000), 2000), batch_size=1, shuffle=True))).to(device)
                    future_observations = smc_prediction(guide, model, data.sub_sequence(500), 1000, 1500, 2)
                    future_capacity = future_observations[:, :, -1].cpu()
                    for epoch_idx in range(future_capacity.shape[1]):
                        plt.plot(range(500, 2000), future_capacity[:, epoch_idx])
                    plt.plot(data.observations[0, :, -1].cpu())
                    summary_writer.add_figure('filtering/test', plt.gcf(), step)
                    plt.close()

                summary_writer.flush()

        torch.save(
            dict(guide=guide.state_dict(),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=step,
                num_particles=num_particles,
                log_dir=summary_writer.log_dir), os.path.join(checkpoint_dir, str(step) + '.pt'))


if __name__ == '__main__':
    fire.Fire(Trainer)

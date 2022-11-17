############################################################################
# Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail   #
############################################################################

import copy
import torch
import random
import os,sys
import numpy as np
import torch.nn as nn

from agents.models import A2Cnet
from agents.base_agent import base_agent

# linear exploration schedule
class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)


class a2c_agent(base_agent):
    def __init__(self, env, args, device, name, victim):
        base_agent.__init__(self, env, args, device, name, victim)
        self.env = env
        self.args = args 
        self.device = device
        # define the network
        self.net = A2Cnet(self.env.action_space.n)
        self.net.to(self.device)

        self.value_loss_coef = args.policy_value_loss_coef
        self.entropy_coef = args.policy_entropy_coef
        self.max_grad_norm = args.policy_max_grad_norm

        self.optimizer = torch.optim.RMSprop(self.net.parameters(), args.a2c_lr, eps=args.a2c_eps, alpha=args.a2c_alpha)


    def select_action(self, inputs, rnn_hxs, masks, log_probs=False, deterministic=True):
        value, actor_features, rnn_hxs = self.net(inputs, rnn_hxs, masks)
        dist = self.net.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()

        action_log_probs = dist.log_probs(action)

        if log_probs:
            action_numbers = torch.linspace(0,self.env.action_space.n-1,steps=self.env.action_space.n).to(self.device)
            complete_action_log_probs = [dist.log_probs(ac.to(self.device).reshape(1,1)) for ac in action_numbers]
            complete_action_log_probs = torch.cat(complete_action_log_probs).reshape(1, len(action_numbers))
            return value, action, action_log_probs, complete_action_log_probs
        else:
            return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.net(inputs, rnn_hxs, masks)
        return value

    def update_agent(self, j, rollouts, advmask=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, 1),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.net.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        if advmask is not None:
            values, action_log_probs, dist_entropy, _ = self.evaluate_actions(
            torch.clamp(rollouts.obs[:-1].view(-1, *obs_shape)-advmask, 0.0, 255.0),
            rollouts.recurrent_hidden_states[0].view(
                -1, 1),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()

            nn.utils.clip_grad_norm_(self.net.parameters(),
                                        self.max_grad_norm)

            self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.net(inputs, rnn_hxs, masks)
        dist = self.net.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

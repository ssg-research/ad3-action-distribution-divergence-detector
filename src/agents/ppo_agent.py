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
from agents.median_pool import MedianPool2d
from rl_utils.utils import update_linear_schedule


class ppo_agent(base_agent):
    def __init__(self, env, args, device, name, victim):
        base_agent.__init__(self, env, args, device, name, victim)
        self.env = env
        self.args = args
        self.device = device

        # define the network
        self.net = A2Cnet(self.env.action_space.n)
        self.net.to(self.device)

        self.initial_clip_param = args.ppo_clip_param
        self.clip_param = args.ppo_clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.ppo_num_mini_batch

        self.value_loss_coef = args.policy_value_loss_coef
        self.entropy_coef = args.policy_entropy_coef
        self.max_grad_norm = args.policy_max_grad_norm
        self.use_clipped_value_loss = args.ppo_use_clipped_value_loss

        self.use_linear_lr_decay = args.use_linear_lr_decay

        self.initial_lr = args.ppo_lr

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.initial_lr, eps=args.ppo_eps)


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
        num_steps, num_processes, _ = rollouts.rewards.size()

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            if(self.use_linear_lr_decay):
                lr = update_linear_schedule(self.optimizer, e, self.ppo_epoch, self.initial_lr)
                self.clip_param = self.initial_clip_param *lr

            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.net(inputs, rnn_hxs, masks)
        dist = self.net.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

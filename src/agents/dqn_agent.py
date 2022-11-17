import copy
import torch
import random
import os,sys
import numpy as np
import torch.nn as nn

#from scipy.special import softmax
from agents.base_agent import base_agent
from agents.models import DQNnet
from rl_utils.experience_replay import replay_buffer
from agents.median_pool import MedianPool2d

# Linear exploration schedule
class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)

class dqn_agent(base_agent):
    def __init__(self, env, args, device, name, victim):
        base_agent.__init__(self, env, args, device, name, victim)
        self.env = env
        self.args = args 
        self.device = device
        # define the network
        self.net = DQNnet(self.env.action_space.n, self.args.use_dueling)
        # copy the self.net as the 
        self.target_net = copy.deepcopy(self.net)
        # make sure the target net has the same weights as the network
        self.target_net.load_state_dict(self.net.state_dict())
        self.net.to(self.device)
        self.target_net.to(self.device)
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        # define the replay memory
        self.buffer = replay_buffer(self.args.buffer_size)
        # define the linear schedule of the exploration
        self.explore_eps = self.args.init_ratio
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), \
                                                    self.args.final_ratio, self.args.init_ratio)

    # select actions
    def select_action(self, obs, explore_eps=0.0):
        with torch.no_grad():
            action_value = self.net(obs)
        action = torch.argmax(action_value) if random.random() > explore_eps else torch.randint(0, action_value.shape[1]-1,size=(1,))
        return action.reshape(1,1), action_value

    def remember(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs.to(self.device), action.to(self.device), reward.to(self.device), next_obs.to(self.device), float(done))

    def update_agent(self, timestep, advmask=None):
        td_loss = np.inf
        if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
            # start to sample the samples from the replay buffer
            batch_samples = self.buffer.sample(self.args.batch_size)
            td_loss = self._update_network(batch_samples, advmask)
        if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
            # update the target network
            self.target_net.load_state_dict(self.net.state_dict())
        return td_loss

    # update the network
    def _update_network(self, samples, advmask=None):
        obses, actions, rewards, obses_next, dones = samples
        
        # convert the data to tensor
        obses = torch.cat(obses) 
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        obses_next = torch.cat(obses_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).to(self.device)
        # calculate the target value
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                q_value_ = self.net(obses_next)
                action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                target_action_value = self.target_net(obses_next)
                target_action_max_value = target_action_value.gather(1, action_max_idx)
            else:
                target_action_value = self.target_net(obses_next)
                target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
        # target
        expected_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(obses)
        real_value = action_value.gather(1, actions)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if advmask is not None:
            if np.random.random_sample() < 0.5:
                for mask in advmask:
                    obses = torch.clamp(obses+mask, 0.0, 255.0)
                    obses_next = torch.clamp(obses_next+mask, 0.0, 255.0)
                    with torch.no_grad():
                        # if use the double network architecture
                        if self.args.use_double_net:
                            q_value_ = self.net(obses_next)
                            action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                            target_action_value = self.target_net(obses_next)
                            target_action_max_value = target_action_value.gather(1, action_max_idx)
                        else:
                            target_action_value = self.target_net(obses_next)
                            target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
                    # target
                    expected_value = rewards + self.args.gamma * target_action_max_value * dones
                    # get the real q value
                    action_value = self.net(obses)
                    real_value = action_value.gather(1, actions)
                    loss = (expected_value - real_value).pow(2).mean()
                    # start to update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        return loss.item()
        


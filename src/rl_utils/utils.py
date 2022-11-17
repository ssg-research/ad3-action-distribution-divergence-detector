# Authors: Shelly Wang, Buse G. A. Tekgul
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import torch
from baselines.common.vec_env.vec_normalize import VecNormalize

# set random seeds for the pytorch, numpy and random
def set_seeds(args, rank=0):
    # set seeds for the numpy
    np.random.seed(args.seed + rank)
    # set seeds for the random.random
    random.seed(args.seed + rank)
    # set seeds for the pytorch
    torch.manual_seed(args.seed + rank)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards 
        self.buffer = [0.0]
        self._episode_length = 1
    
    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)
    
    @property
    def mean(self):
        return np.mean(self.buffer)
    
    # get the length of total episodes
    @property 
    def num_episodes(self):
        return self._episode_length


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def make_files_and_paths(args):
    adversary = args.adversary
    if args.save_frames:
        image_path  =  args.env_name + 'images/'
    if args.adversarial_retraining_defense == True:
        addname = 'adv'
    else:
        addname = ''

    if args.adversary == 'obs_fgsm_wb':
        noise_path = args.env_name + 'noise'+ addname + '/' + adversary+'_' + args.training_frames_type + str(args.training_frames) \
                     + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                     + '_obs_freq' + str(args.attack_frequency_frame) \
                     + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        reward_path = args.env_name + 'rewards'+ addname + '/' + adversary+'_' + args.training_frames_type + str(args.training_frames)\
                      + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                      + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                      + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        reward_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_' + args.training_frames_type + str(
            args.training_frames) \
                      + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                      + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                      + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "reward_per_steps-{}" + '.npy'

        detection_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_' + args.training_frames_type + str(
            args.training_frames) \
                      + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                      + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                      + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "detection_per_steps_game-{}" + '.npy'
    else:
        noise_path  = args.env_name + 'noise' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
					+ '_obs_freq' + str(args.attack_frequency_frame) \
					+ '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        reward_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
					+ '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                                    	+ '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        reward_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                      + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                      + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "reward_per_steps-{}" + '.npy'
        detection_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
                      + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
                      + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "detection_per_steps-{}" + '.npy'

    #if args.adversary == 'uap_single_frame':
    #    training_data_path = args.env_name + 'train_dataset/uap' + '_' + args.victim_agent_mode + '_train_data'
    #else:
    #    training_data_path = args.env_name + 'train_dataset/' + adversary + '_' + args.victim_agent_mode + '_train_data'
    training_data_path = args.env_name + 'train_dataset'+ addname + '/' + args.victim_agent_mode + '_train_data'

    if not os.path.isdir(args.env_name + 'rewards'+ addname + '/'):
        os.makedirs(args.env_name + 'rewards'+ addname + '/')
    if not os.path.isdir(args.env_name + 'train_dataset'+ addname + '/'):
        os.makedirs(args.env_name + 'train_dataset'+ addname + '/')
    if not os.path.isdir(args.env_name + 'rewards'+ addname + '/'):
        os.makedirs(args.env_name + 'rewards'+ addname + '/')
    if not os.path.isdir(args.env_name + 'noise'+ addname + '/'):
        os.makedirs(args.env_name + 'noise'+ addname + '/')

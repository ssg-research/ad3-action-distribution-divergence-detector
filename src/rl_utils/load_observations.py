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

import os,sys
import torch
import glob
from rl_utils.atari_wrapper import make_vec_envs

def make_files_and_paths(args):
   # adversary = args.adversary
    if args.save_frames:
        image_path  =  args.env_name + 'images/'
    #if args.adversarial_retraining_defense == True:
    #    addname = 'adv'
    #else:
    #    addname = ''

    """if args.adversary == 'obs_fgsm_wb':
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
        #noise_path  = args.env_name + 'noise' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
		#			+ '_obs_freq' + str(args.attack_frequency_frame) \
		#			+ '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        #reward_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
		#			+ '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
        #                            	+ '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + '.npy'
        #reward_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
        #              + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
        #              + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "reward_per_steps-{}" + '.npy'
        #detection_step_path = args.env_name + 'rewards' + addname + '/' + adversary + '_victim' + args.victim_agent_mode + '_attacker' + args.attacker_agent_mode \
        #              + '_attack_ratio' + str(args.attack_ratio) + '_obs_freq' + str(args.attack_frequency_frame) \
        #              + '_ac' + str(args.action_threshold) + '_eps' + str(args.eps) + "detection_per_steps-{}" + '.npy'
        """
    #if args.adversary == 'uap_single_frame':
    #    training_data_path = args.env_name + 'train_dataset/uap' + '_' + args.victim_agent_mode + '_train_data'
    #else:
    #    training_data_path = args.env_name + 'train_dataset/' + adversary + '_' + args.victim_agent_mode + '_train_data'
    #training_data_path = args.env_name + 'train_dataset'+ addname + '/' + args.victim_agent_mode + '_train_data'

    #if not os.path.isdir(args.env_name + 'rewards'+ addname + '/'):
    #    os.makedirs(args.env_name + 'rewards'+ addname + '/')
    #if not os.path.isdir(args.env_name + 'train_dataset'+ addname + '/'):
    #    os.makedirs(args.env_name + 'train_dataset'+ addname + '/')
    #if not os.path.isdir(args.env_name + 'rewards'+ addname + '/'):
    #    os.makedirs(args.env_name + 'rewards'+ addname + '/')
    #if not os.path.isdir(args.env_name + 'noise'+ addname + '/'):
    #    os.makedirs(args.env_name + 'noise'+ addname + '/')

    #return image_path, noise_path, reward_path, training_data_path, reward_step_path, detection_step_path

def load_training_files(training_path):
    file_list_x = []
    #file_list_y = []
    file_list_c = []
    file_total_observations = training_path + "_total_observations.pt"

    for file in sorted(glob.glob(training_path + "_X_batch_" + "*.pt")):
        print(file)
        file_list_x.append(file)
    #for file in sorted(glob.glob(training_path + "_Y_batch_" + "*.pt")):
    #    print(file)
    #    file_list_y.append(file)
    for file in sorted(glob.glob(training_path + "_c_batch_" + "*.pt")):
        print(file)
        file_list_c.append(file)

    print("There are {} batches in the training set".format(len(file_list_x)))

    #return file_list_x, file_list_y, file_list_c, file_total_observations
    return file_list_x, file_list_c, file_total_observations

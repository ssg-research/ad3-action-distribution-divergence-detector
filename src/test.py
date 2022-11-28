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

import random
import time
import torch
import logging
import numpy as np
import scipy

from agents.dqn_agent import dqn_agent
from agents.a2c_agent import a2c_agent
from agents.ppo_agent import ppo_agent
from detection_module.DetectionModule import DetectionModule

from rl_utils.utils import set_seeds
from rl_utils.atari_wrapper import make_vec_envs
from agents import action_conditional_video_prediction as acvp

def attack(env, victim_agent_mode, obs, device, adversary=None):
    if adversary!= "none":
        #load the previously saved adversarial mask for the correct game, agent and adversary type
        path_to_mask = "universal_noise_masks/" + env + "/" + "victim_" + victim_agent_mode + "_" + adversary + ".npy"
        advmask = torch.load(path_to_mask, map_location=device)["advmask"]
        obs_adv = torch.clamp(obs+advmask, 0.0, 255.0)
        return obs_adv
    obs_adv = obs.clone()
    return obs_adv

def act(obs, obs_adv, agent, recurrent_hidden_states, masks, args,
        predicted_action=None, predicted_dist=None, detection_alarm=False):
    change_in_action = 0

    if args.victim_agent_mode == 'dqn':
        start_a = time.time()
        action, action_distribution = agent.select_action(obs_adv)
        end_a = time.time()
        original_action, _ = agent.select_action(obs)
    elif args.victim_agent_mode == 'a2c' or args.victim_agent_mode == 'ppo':
        start_a = time.time()
        with torch.no_grad():
            _, action, _, action_distribution = agent.select_action(obs_adv, recurrent_hidden_states, masks,
                                                                    log_probs=True)
            end_a = time.time()
            _, original_action, _, action_distribution_o = agent.select_action(obs, recurrent_hidden_states, masks,
                                                                               log_probs=True)

    if args.visual_foresight_defense == True and predicted_dist is not None:

        diff = abs(action_distribution - predicted_dist).sum()/len(action_distribution)

        if diff > 0.01:
            action = predicted_action


    # Change action distributions to softmax
    if action.item() != original_action.item():
        change_in_action = 1

    return action, original_action, action_distribution, change_in_action, (end_a - start_a)


def calculate_games_won(args, reward_arr, detection_arr):
    # if game terminated then you don't loose
    loose_games = [int(x < 0) for x in reward_arr]

    defense_loose_arr = []
    if args.detection_method == "KL":
        for i in range(len(reward_arr)):
            # the lost game
            defense_loose_arr.append(int(loose_games[i] == 1 and detection_arr[i] == 0))

    return loose_games, defense_loose_arr


def setup_test_agent_env(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Environment, create
    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, 'output/env_logs', device=device,
                        allow_early_resets=args.allow_early_resets)
    # Environment, set seeds
    set_seeds(args)
    # Victim, load a trained agent
    model_name = 'model.pt'
    victim = True
    if args.victim_agent_mode == 'dqn':
        agent = dqn_agent(env, args, device, model_name, victim)
    elif args.victim_agent_mode == 'a2c':
        agent = a2c_agent(env, args, device, model_name, victim)
    elif args.victim_agent_mode == 'ppo':
        agent = ppo_agent(env, args, device, model_name, victim)
    if args.load_from is not None:
        agent.model_path = args.load_from
    agent.net.load_state_dict(torch.load(agent.model_path, map_location=lambda storage, loc: storage))
    # Initilaizations related to actor-critic models
    recurrent_hidden_states = torch.zeros(1, 1)
    masks = torch.zeros(1, 1)

    return agent, device, env, masks, recurrent_hidden_states


def test(args):
    #image_path, noise_path, reward_path, training_data_path, reward_step_path, detection_step_path = make_files_and_paths(
    #    args)
    #detection_step_path = args.env_name + '_rewards' + '/' + 'victim_' + args.victim_agent_mode + '_attacker_' + \
    #                     '_ac' + str(args.action_threshold) + "_detection_per_steps_{}" + '.npy'
    logging.basicConfig(filename=(args.env_name + '_' + args.victim_agent_mode + '_' + 'measurements.log'),
                        level=logging.DEBUG)
    # fstats = open(args.env_name + '_' + args.victim_agent_mode + '_' + 'distancestats.txt','a+')

    agent, device, env, masks, recurrent_hidden_states = setup_test_agent_env(args)

    # setup_detection module
    detect_mod = DetectionModule(env.action_space.n, args)

    if args.detection_method == "KL":
        if not detect_mod.check_train_model():
            print("Cannot find trained model for detection module. "
                  "First run with argument --detection-method-train to get a train detection model.")
            return None

        else:
            detect_mod.load_train_model()

    games_to_play = args.total_game_plays + args.attacker_game_plays

    # Measurements, define episodic statistics (rewards, time spent by attacker, ...
    # ... time spent by agent, change in agent's decisions 
    total_rewards = np.zeros(games_to_play - args.attacker_game_plays)
    time_attack = np.zeros(games_to_play - args.attacker_game_plays)
    time_agent = np.zeros(games_to_play - args.attacker_game_plays)
    action_change = np.zeros(games_to_play - args.attacker_game_plays)
    total_alarms = np.zeros(games_to_play - args.attacker_game_plays)
    total_action_change = 0

    # load the adversarial defense, if any
    if args.visual_foresight_defense == True or args.detection_method == "VF":
        vf = acvp.Network(env.action_space.n).to(device)
        checkpoint = torch.load(agent.defense_path + args.victim_agent_mode + "_acvp.pth",
                                map_location=lambda storage, loc: storage)
        vf.load_state_dict(checkpoint['model_state_dict'])
        vf.eval()
        action_one_hot = torch.zeros((env.action_space.n,)).to(device)
        mean_obs = checkpoint['mean_obs'].to(device)
        defense_time = []

    # Environment, begin test
    frame_idx_total = 0
    predicted_action = None
    predicted_dist = None

    print("start testing. Total game plays: {}".format(games_to_play))

    for game_id in range(games_to_play):
        obs = env.reset()
        frame_idx_ingame = 0

        cur_reward = 0
        alarm = False
        game_reward_per_step = []
        game_detection_per_step = []
        defense_time = []
        is_attacking = False
        alternate_attacked_frame_count = 0
        action_change_count = 0

        while True:
            # Environment, render (i.e., show gameplay)
            if args.render:
                env.render()

            # Adversary, generate perturbation mask if necessary
            # adv.generate(obs, game_id, frame_idx_ingame, args.attacker_game_plays)
            #if is_attacking:
            #    obs_adv, time_attack_val = adv.attack(obs, frame_idx_ingame, agent)
            #else:
            obs_adv = attack(args.env_name, args.victim_agent_mode, obs, device, adversary=args.adversary)

            # Victim, select actions
            action, original_action, action_distribution, action_change_val, time_agent_val = act(obs, obs_adv,
                                                                                 agent, recurrent_hidden_states, masks,
                                                                                 args,
                                                                                 predicted_action=predicted_action,
                                                                                 predicted_dist=predicted_dist,
                                                                                 detection_alarm=alarm)
            action_change_count += action_change_val
            time_agent[game_id - args.attacker_game_plays] += time_agent_val

            # if args.env_name == "Breakout" and frame_idx_ingame > 3000 and frame_idx_ingame % 100 == 0:
            #     r = random.randint(0, env.action_space.n - 1)
            #     action[0][0] = r
            #     original_action[0][0] = r

            if args.detection_method != "none" and game_id >= args.attacker_game_plays:
                start = time.time()
                score, alarm = detect_mod.process_step(original_action.item(), frame_idx_ingame,
                                                       game_id - args.attacker_game_plays, action_change_val,
                                                       action_distribution,
                                                       predicted_action, predicted_dist)
                end = time.time()
                defense_time.append(end - start)

            # Victim, execute actions
            next_obs, reward, done, info = env.step(action)

            # Visual foresight prediction for !next state!
            if args.visual_foresight_defense == True and args.victim_agent_mode == 'dqn':
                with torch.no_grad():
                    action_one_hot[action_one_hot!=0] = 0
                    action_one_hot[action] = 1
                    if (args.adversary == 'fgsm') or (args.adversary == 'none') or (game_id < args.attacker_game_plays):
                        next_obs_predict = next_obs.clone()
                    else:
                        next_obs_predict = adv.attack(next_obs, frame_idx_ingame, agent)[0].clone()
                    start = time.time()

                    temp = acvp.post_process(vf.predict(acvp.pre_process(next_obs_predict[:,:3,:,:], mean_obs), action_one_hot), mean_obs)
                    next_obs_predict[:,-1,:,:] = torch.round(temp.clone().detach())
                    predicted_action, predicted_dist = agent.select_action(torch.ceil(next_obs_predict))
                    end = time.time()
                    defense_time.append(end-start)

            if args.save_detection_scores and game_id >= args.attacker_game_plays:
                cur_reward += reward.item()
                game_reward_per_step.append(cur_reward)
                game_detection_per_step.append(alarm)

            # Environment, prepare for next obs, print if game ends
            obs = next_obs.clone()
            frame_idx_total += 1
            frame_idx_ingame += 1

            if 'episode' in info[0].keys():
                if game_id >= args.attacker_game_plays:
                    total_rewards[game_id - args.attacker_game_plays] = info[0]['episode']['r']
                    action_change[game_id - args.attacker_game_plays] = action_change_count / frame_idx_ingame
                    total_action_change += action_change_count
                    time_agent[game_id - args.attacker_game_plays] = time_agent[game_id - args.attacker_game_plays] / frame_idx_ingame
                if args.detection_method != "none" and game_id >= args.attacker_game_plays:
                    total_alarms[game_id - args.attacker_game_plays] = detect_mod.is_alarmed()
                    detect_mod.save_and_clean(info[0]['episode']['r'], detect_mod.eps)
                print('Game id: {}, score: {}, total number of state-action pairs: {}'.format(game_id+1,
                                                                                              info[0]['episode']['r'],
                                                                                              frame_idx_ingame))
                break

    env.close()

    # Measurements, save statistics to files, log and print
    # TODO: Some results are shown as zero, check why
    # np.save(reward_path, total_rewards)
    print("Average reward: {:.2f} std: {:.2f}".format(np.mean(total_rewards), np.std(total_rewards)))
    print("Tmax: {:.6f} secs".format(1.0/60.0 - np.mean(time_agent)))
    logging.info("Tmax: {:.6f} secs".format(1.0/60.0 - np.mean(time_agent)))
    logging.info("Average action change rate: {:.3f}".format(100 * np.sum(action_change) / frame_idx_total))
    logging.info("Average reward: {:.2f}".format(np.mean(total_rewards)))
    logging.info("Average reward variance: {:.2f}".format(np.std(total_rewards)))

    loose_game, det_loose_game = calculate_games_won(args, total_rewards, total_alarms)

    if args.detection_method != "none":
        logging.info("Detection method: {}".format(args.detection_method))
        logging.info("Reward arr: {}".format(total_rewards))
        logging.info("Detection arr: {}".format(total_alarms))
        logging.info("Action change arr: {}".format(action_change))

        print("Average alarm: {}".format(np.mean(total_alarms)))
        logging.info("Average alarm: {}".format(np.mean(total_alarms)))
        print("Average time for detection module: {:.6f} secs".format(100*np.mean(defense_time)))
        logging.info("Average time for detection module: {:.6f} secs".format(100* np.mean(defense_time) ))
        logging.info("Average time variance for detection module: {:.6f} secs".format(100* np.std(defense_time)))

        print("Average game lost (with no detection): {}".format(np.mean(loose_game)))
        print("Average game lost (with detection): {}".format(np.mean(det_loose_game)))

        logging.info("Average game lost (with no detection): {}".format(np.mean(loose_game)))
        logging.info("Average game lost (with detection): {}".format(np.mean(det_loose_game)))

    if args.visual_foresight_defense:
        logging.info("Visual Foresight: ON")
        print("Average game lost (with no detection): {}".format(np.mean(loose_game)))
        logging.info("Average game lost (with no detection): {}".format(np.mean(loose_game)))

    if args.visual_foresight_defense == True:
        print("Average time for visual foresight defense: {} secs".format(100*np.mean(defense_time)))
        logging.info("Average time for visual foresight defense: {} secs".format(100*np.mean(defense_time)))
        logging.info("Average time variance for visual foresight defense: {} secs".format(100*np.std(defense_time)))

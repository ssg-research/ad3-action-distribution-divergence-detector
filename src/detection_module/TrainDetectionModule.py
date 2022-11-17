from datetime import time
import random

import numpy as np

from detection_module.DetectionModule import DetectionModule
from test import act, setup_test_agent_env


def train_detection_module(args):
    print("Training detection module for game {}".format(args.env_name))

    agent, device, env, masks, recurrent_hidden_states = setup_test_agent_env(args)

    # setup_detection module
    detect_mod = DetectionModule(env.action_space.n, args)
    games_to_play = (args.detection_game_plays * 2)

    # first play multiple games to collect "normal" agent behaviour
    for game_id in range(args.detection_game_plays):
        obs = env.reset()
        frame_idx_ingame = 0
        while True:
            # Environment, render (i.e., show gameplay)
            if args.render:
                env.render()

            obs_adv = obs.clone()
            # Victim, select actions
            action, _, action_distribution, action_change_val, time_agent_val = act(obs, obs_adv, agent,
                                                                                 recurrent_hidden_states, masks, args)

            # add randomness to Breakout at the end of game so that the agent would not be stuck
            # in an infinite loop
            if args.env_name == "Breakout" and frame_idx_ingame > 3000 and frame_idx_ingame % 100 == 0:
                r = random.randint(0, env.action_space.n - 1)

                action[0][0] = r

            # Victim, execute actions
            next_obs, reward, done, info = env.step(action)
            masks.fill_(0.0 if done else 1.0)

            detect_mod.train_module(action.item())

            # Environment, prepare for next obs, print if game ends
            obs = next_obs.clone()
            frame_idx_ingame += 1

            if 'episode' in info[0].keys():
                print('Game id: {}, score: {}, total number of state-action pairs: {}'
                      .format(game_id, info[0]['episode']['r'], frame_idx_ingame))
                break

    detect_mod.save_train_model()
    detect_mod.clean()
    detect_mod.load_train_model()

    # play more games to collect anomaly scores between normal games
    for game_id in range(games_to_play):
        obs = env.reset()
        frame_idx_ingame = 0

        while True:
            # Environment, render (i.e., show gameplay)
            if args.render:
                env.render()

            obs_adv = obs.clone()

            # Victim, select actions
            action, _, action_distribution, action_change_val, time_agent_val = act(obs, obs_adv, agent,
                                                                                 recurrent_hidden_states, masks,
                                                                                 args)

            # add randomness to Breakout at the end of game so that the agent would not be stuck
            if args.env_name == "Breakout" and frame_idx_ingame > 2000:
                r = random.randint(0, env.action_space.n - 1)
                action[0][0] = r

            # Victim, execute actions
            next_obs, reward, done, info = env.step(action)

            if args.detection_method != "none":
                score, alarm = detect_mod.process_step(action.item(), frame_idx_ingame, game_id, action_change_val)

            # Environment, prepare for next obs, print if game ends
            obs = next_obs.clone()
            frame_idx_ingame += 1

            if 'episode' in info[0].keys():
                if args.detection_method != "none":
                    detect_mod.save_and_clean(info[0]['episode']['r'], args.eps)
                print('Game id: {}, score: {}, total number of state-action pairs: {}'.format(game_id,
                                                                                              info[0]['episode'][
                                                                                                  'r'],
                                                                                              frame_idx_ingame))
                break

    env.close()

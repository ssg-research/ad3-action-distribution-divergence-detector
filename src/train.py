import gym
import time
import torch
import os, sys
import numpy as np
from collections import deque
from datetime import datetime


from agents.dqn_agent import dqn_agent
from agents.a2c_agent import a2c_agent
from agents.ppo_agent import ppo_agent
from agents.storage import RolloutStorage
import agents.action_conditional_video_prediction as acvp
from rl_utils.atari_wrapper import create_single_env, make_vec_envs
from rl_utils.utils import reward_recorder, set_seeds


def train(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.visual_foresight_defense == True:
        env = make_vec_envs(args.env_name, args.seed+1, 1, args.gamma, device, allow_early_resets=args.allow_early_resets)
        set_seeds(args, 1)
        args.game_mode = 'test'
        if args.victim_agent_mode == 'dqn':
            model_name = "model.pt"
            victim = True
            agent = dqn_agent(env, args, device, model_name, victim)
        elif args.victim_agent_mode == 'a2c':
            agent = a2c_agent(env, args, device)
        elif args.victim_agent_mode == 'ppo':
            agent = ppo_agent(env, args, device)
        if os.path.isfile(agent.model_path): 
            acvp.train(args, env, agent)
        else:
            raise Exception("There should be a trained agent in the model path")
    else:
        # no defense is deployed
        if args.victim_agent_mode == 'dqn':
            env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, 'output/env_logs', device, allow_early_resets=args.allow_early_resets)
            set_seeds(args)
            dqn_train(args, env, device)
        elif args.victim_agent_mode == 'a2c' or args.victim_agent_mode == 'ppo':
            env = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, 'output/env_logs', device, False)
            set_seeds(args)
            policy_train(args, env, device)  

def dqn_train(args, env, device, agent=None, advmask=None):
    #agent = dqn_agent(env, args, device)
    model_name = "model.pt"
    victim = True
    agent = dqn_agent(env, args, device, model_name, victim)

    # the episode reward
    episode_rewards = deque(maxlen=100)
    obs = env.reset()
    td_loss = 0.0
    timestep = 0
    start = time.time()
    # start to learn 
    for timestep in range(args.total_timesteps): 
        explore_eps = agent.exploration_schedule.get_value(timestep)   
        # select actions
        action, _ = agent.select_action(obs, explore_eps)
        # excute actions
        next_obs, reward, done, infos = env.step(action)
        if args.render:
            env.render()
        # trying to append the samples
        agent.remember(obs, action, np.sign(reward), next_obs, done)
        obs = next_obs.clone()
        # add the rewards
        #episode_reward.add_rewards(reward)
        td_loss =agent.update_agent(timestep, advmask)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                if timestep % args.display_interval == 0 and len(episode_rewards) > 1:
                    end = time.time()
                    print("{} updates has been done out of total {} of updates in training, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(timestep, args.total_timesteps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards)))
        # save for every interval-th episode or for the last epoch
        if (timestep% (args.dqn_save_interval) == 0) and args.save_dir != "":
            torch.save(agent.net.state_dict(), agent.model_path + model_name) 
    # close the environment and save the last trained state of the policy
    print("Saving the final state of the policy...")
    torch.save(agent.net.state_dict(), agent.model_path + model_name)
    env.close()

def policy_train(args, envs, device, agent=None, advmask=None):
    model_name = "model.pt"
    victim = 1
    if args.victim_agent_mode == 'a2c':
        agent = a2c_agent(envs, args, device, model_name, victim)
    else:
        agent = ppo_agent(envs, args, device, model_name, victim)

    rollouts = RolloutStorage(args.policy_num_steps, args.num_processes,
                            (4,84,84), envs.action_space,1)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device) 

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(args.total_timesteps) // args.policy_num_steps // args.num_processes
    for j in range(num_updates):

        for step in range(args.policy_num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.select_action(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, use_proper_time_limits=False)

        value_loss, action_loss, dist_entropy = agent.update_agent(j, rollouts=rollouts, advmask=advmask)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % (args.policy_save_interval) == 0
                or j == num_updates - 1) and args.save_dir != "":
            torch.save(agent.net.state_dict(), agent.model_path + model_name) 

        if j % args.policy_log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.policy_num_steps
            end = time.time()
            print("{} updates has been done out of total {} of updates in training, \nFPS {}, Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, args.total_timesteps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss))
    # close the environment and save the last trained state of the policy
    print("Saving the final state of the policy...")
    torch.save(agent.net.state_dict(), agent.model_path + model_name)
    env.close()  
    



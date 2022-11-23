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

import argparse
import atari_py


from test import test
from train import train
from detection_module.TrainDetectionModule import train_detection_module

def get_args():
    parse = argparse.ArgumentParser()
    available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
    parse.add_argument('--env-name', default='Pong', help='Choose from available games: ' + str(available_games) + ". Default is 'breakout'.")
    parse.add_argument('--env-type', type=str, default='atari', help='the environment type')
    parse.add_argument('--game-mode', default='train', help="Choose from available modes: train, test, Default is 'train'.")
    parse.add_argument('--victim-agent-mode', default='dqn', help="Choose from available RL algorithms: dqn, a2c, ppo, Default is 'dqn'.")
    parse.add_argument('--load-from', default=None, help="Load a previously saved agent")
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-processes',type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parse.add_argument('--cuda', type=bool, default=True, help='if use the gpu')
    parse.add_argument('--grad-norm-clipping', type=float, default=10, help='the gradient clipping')
    parse.add_argument('--total-timesteps', type=int, default=int(2e7), help='the total timesteps to train network')   #int(2e7)
    parse.add_argument('--total-game-plays', type=int, default=int(10), help='the total number of independent game plays in test time')
    #parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--display-interval', type=int, default=5, help='the display interval')
    parse.add_argument('--render', type=bool, default=False, help='render environment')
    parse.add_argument('--save-frames', type=bool, default=True, help= 'save frames for attack or no attack conditions')
    parse.add_argument('--allow-early-resets', type=bool, default=False, help= 'allows early resets in game. !!In Freeway, you should allow early resets.')

    # DQN related arguments
    parse.add_argument('--batch-size', type=int, default=32, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--init-ratio', type=float, default=1, help='the initial exploration ratio')
    parse.add_argument('--exploration_fraction', type=float, default=0.1, help='decide how many steps to do the exploration')
    parse.add_argument('--final-ratio', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--buffer-size', type=int, default=10000, help='the size of the buffer')
    parse.add_argument('--learning-starts', type=int, default=10000, help='the frames start to learn')
    parse.add_argument('--train-freq', type=int, default=4, help='the frequency to update the network')
    parse.add_argument('--target-network-update-freq', type=int, default=1000, help='the frequency to update the target network')
    parse.add_argument('--use-double-net', action='store_true', help='use double dqn to train the agent')
    parse.add_argument('--use-dueling', action='store_false', help='use dueling to train the agent')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--dqn-save-interval',type=int,default=100,help='save interval, one save per n updates (default: 100)')

    # policy agent related arguments
    parse.add_argument('--policy-value-loss-coef',type=float,default=0.5,help='value loss coefficient (default: 0.5)')
    parse.add_argument('--policy-entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parse.add_argument('--policy-max-grad-norm',type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parse.add_argument('--policy-num-steps', type=int, default=5,help='number of forward steps in Policy agents (default: 5)')
    parse.add_argument('--policy-log-interval',type=int,default=10,help='log interval, one log per n updates (default: 10)')
    parse.add_argument('--policy-save-interval',type=int,default=100,help='save interval, one save per n updates (default: 100)')
    parse.add_argument('--use-gae', type=bool, default=False, help='use generalized advantage estimation')
    parse.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')

    # A2C related parameters
    parse.add_argument('--a2c-lr', type=float, default=7e-4, help='a2c learning rate (default: 7e-4)')
    parse.add_argument('--a2c-eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parse.add_argument('--a2c-alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')

    #TODO: deal with it.. do we still use this?
    parse.add_argument('--a2c-eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')

    # PPO related arguments
    parse.add_argument('--ppo-lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parse.add_argument('--ppo-eps', type=float, default=1e-5, help='Adam optimizer epsilon (default: 1e-5)')
    parse.add_argument('--ppo-clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parse.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parse.add_argument('--ppo-num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parse.add_argument('--ppo-use-clipped-value-loss', type=bool, default=True, help='PPO use cliiped value loss (default: True)')
    parse.add_argument('--use-linear-lr-decay', type=bool, default=False, help='Use linear lr decay in PPO (default: False)')

    # Defense related arguments
    parse.add_argument('--visual-foresight-defense', type=bool, default=False, help="if true, action conditional video prediction is used as a visual foresight defense")
    parse.add_argument('--adversarial-retraining-defense', type=bool, default=False, help="if true, action conditional video prediction is used as a visual foresight defense")
    parse.add_argument('--defense-game-plays', type=int, default=int(100), help='the total number of independent game plays in training time for visual foresight defense')

    # Detection related arguments
    parse.add_argument('--detection-method', type=str, default="none", help="Use detection module: none, KL, VF. Default is 'none'")
    parse.add_argument("--skipped-frames", type=int, default=400, help="Number of frames skipped for calculating threshold")
    parse.add_argument("--percentile", type=int, default=100, help="The percentile of max anomaly score from training data to use as threshold. Default: 90.")
    parse.add_argument("--alarm-percentage", type=float, default=float(0.9), help="The perecntage of steps in queue above threshold to raise an alarm. Default: 0.9 .")
    parse.add_argument("--queue-size", type=int, default=200, help="The size of the queue to store the steps that are above or below threshold. Default: 100.")
    parse.add_argument('--detection-game-plays', type=int, default=3, help='the total number of independent game plays as get baseline action distribution. Default: 3.')
    parse.add_argument("--save-detection-scores", action='store_true', help="Store values of games played and the detection alarm steps.")
    parse.add_argument('--use-saved-game', action='store_true', help="Use saved game data to run detection environment.")
    parse.add_argument('--detection-method-train', action='store_true',
                       help="Use saved game data to run detection environment.")

    # Attack related arguments
    parse.add_argument('--adversary', default='none', help="Choose from available modes: none, random, uaps, uapf, oswfu. Default is 'none'.")
    parse.add_argument('--attack-ratio', type=float, default=float(1.0), help="Attack ratio. The percentage of the time steps where we apply adversarial attacks")
    parse.add_argument('--alternate-attack', action='store_true', help="Alternate between attack and no attack with attack duration")
    parse.add_argument('--attacker-game-plays', type=int, default=int(0), help='the total number of independent game plays in training time for attack')

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    # get arguments
    args = get_args()
    print ("Selected environment: " + str(args.env_name))
    print ("Selected victim DRL algorithm: " + str(args.victim_agent_mode).upper())
    print ("Selected mode: " + str(args.game_mode))
    print ("Visual foresight Defense: " + str(args.visual_foresight_defense))
    print ("Detection method: " + str(args.detection_method))

    if args.detection_method != "none":
        print("---Detection Parameters---")
        print("Detection method train: " + str(args.detection_method_train))
        print("Skipped frames: " + str(args.skipped_frames))
        print("Percentile: " + str(args.percentile))
        print("Alarm Percentage: " + str(args.alarm_percentage))
        print("Queue size: " + str(args.queue_size))
        print("Detection game plays: " + str(args.detection_game_plays))

    if args.game_mode == 'train':
        train(args)
    elif args.game_mode == 'test':
        if args.detection_method_train:
            train_detection_module(args)
        else:
            test(args)
    else:
        print ("Unrecognized mode. Use --help")
        exit(1)


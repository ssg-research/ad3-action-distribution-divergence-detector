import collections
import os
import scipy

import numpy as np
import glob

from detection_module.KLDivergence import KLDivergence
from detection_module.KLDivergenceQueue import KLDivergenceQueue
from detection_module.MarkovChain import MarkovChain


def add_paddings(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


class DetectionModule:

    def __init__(self, num_actions, args):
        self.num_game_train = args.detection_game_plays
        self.num_actions = num_actions
        self.technique = args.detection_method
        self.is_training = args.detection_method_train
        self.game_id = -1

        self.use_saved_anomaly_score = False

        self.anomaly_arr = []
        self.anomaly_alarm_arr = []
        self.action_change_arr = []
        self.actions_arr = []
        self.training_list = list()
        self.skipped_frames = args.skipped_frames
        self.queue_size = args.queue_size

        self.reward_arr = []
        # Pre-computed masks in this repo are all within the epsilon 0.01 bound. 
        self.eps = 0.01

        if self.technique == "VF":
            self.threshold = args.detect_threshold
        else:
            self.threshold = 0

        self.percentile = args.percentile
        self.alarm_percentage = args.alarm_percentage

        self.detection_queue = collections.deque(maxlen=self.queue_size+1)

        eps_dir_string = "-" + str(self.eps) if args.adversary != "none" else ""

        if self.is_training:
            root_dir = "detection_model/"
        else:
            root_dir = "anomaly_result/"

        alaternate_dir = ""
        if args.alternate_attack:
            alaternate_dir = "alternate_attack" + str(args.attack_duration) + "/"

        self.base_dir = root_dir + args.env_name + "game_data/" + \
                        args.victim_agent_mode + "/" + args.detection_method + "/"\
                        + args.adversary + eps_dir_string + "/" +\
                        "games_trained-" + str(self.num_game_train) + "/" + alaternate_dir

        self.model_dir = "detection_model/" + args.env_name + "/" + args.victim_agent_mode + "/" +\
                         "games_trained-" + str(self.num_game_train) + "/"

        self.train_model_filename = "{}{}_train_model.npz".format(self.model_dir, self.technique)
        self.training_game_data_filename = "{}{}_train_game_*.npy".format(self.model_dir, self.technique)
        self.training_actions_game_data_filename = "{}{}_train_game_*.npy".format(self.model_dir, self.technique)
        self.anomaly_score_filename = "{}{}_anomalyscore_ratio_{}_game_*.npy".format(self.base_dir, self.technique, args.attack_ratio)
        self.anomaly_alarm_filename = "{}{}_anomalyalarm_ratio_{}_game_*.npy".format(self.base_dir, self.technique, args.attack_ratio)
        self.action_change_filename = "{}{}_actionchange_ratio_{}_game_*.npy".format(self.base_dir, self.technique, args.attack_ratio)
        self.reward_filename = "{}{}_reward_game_*.npy".format(self.base_dir, self.technique)

        self.result_csv = "./anomaly_result/" + args.env_name + "detection_result.csv"
        self.result_base_line = "\n" + ",".join([args.env_name, args.victim_agent_mode, args.adversary,
                                                 str(args.attack_ratio),
                                                 args.detection_method,
                                                 str(self.num_game_train), str(self.skipped_frames),
                                                 str(self.queue_size), str(self.percentile),
                                                 str(self.alarm_percentage)])

        if not os.path.isdir(self.base_dir) and not self.is_training:
            os.makedirs(self.base_dir)

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        if self.technique == "KL":
            self.train = KLDivergence(self.num_actions)
            self.test = KLDivergence(self.num_actions)

    def train_module(self, y):
        self.train.update_action(y)

    def save_train_model(self):
        print("Save trained model to: " + self.train_model_filename)
        self.train.save_model_to_file(self.train_model_filename)

    def check_train_model(self):
        return os.path.exists(self.train_model_filename)

    def load_train_model(self):
        print("Load trained model from: " + self.train_model_filename)
        self.train.load_model_from_file(self.train_model_filename)
        self.test.setup_test(self.train)

        #print("Original train data:")
        #print(self.train.distribution)

        #print("Original test data:")
        #print(self.test.distribution)

        if not self.is_training:
            self.calculate_threshold()

    def save_training_data(self):
        print("Total training games: {}".format(len(self.training_list)))
        for i in range(len(self.training_list)):
            ar = self.training_list[i]
            with open(self.training_game_data_filename.replace("*", str(i)), 'wb') as f:
                np.save(f, np.asarray(ar))

    def setup_detection_threshold(self):
        self.training_list.clear()

        training_filenames = glob.glob(self.training_game_data_filename)
        print("Total training game files:{}".format(training_filenames))

        for file in training_filenames:
            with open(file, 'rb') as f:
                self.training_list.append(np.load(f))

    def calculate_threshold(self):
        self.setup_detection_threshold()

        # process normal game training data. Remove the first x frames
        length_list = list()

        for data in self.training_list:
            length_list.append(len(data))

        max_length = max(length_list)

        temp_data = list()

        for i in range(len(self.training_list)):
            y = self.training_list[i][self.skipped_frames:]
            temp_data.append(add_paddings(list(y), -1, max_length))

        # get the max anomaly score between all game at each step
        normal_data_matrix = np.max(np.stack(temp_data), axis=0)

        # get threshold by taking the x percentile of max anomaly score
        self.threshold = np.percentile(normal_data_matrix, self.percentile)

        print("Percentile: {}  Threshold: {}".format(self.percentile, self.threshold))

    def save_and_clean(self, reward, eps):

        if self.is_training:
            self.training_list.append(self.anomaly_arr)

        if self.technique == "KL":
            self.test.clean()

        if self.is_training:
            filename = self.training_game_data_filename.replace("*", str(self.game_id))
            with open(filename, "wb") as f:
                np.save(f, np.asarray(self.anomaly_arr))
        else:

            with open(self.result_csv, "a") as f:
                alarm_value = max(self.anomaly_alarm_arr)
                line = ",".join([self.result_base_line, str(reward), str(alarm_value), str(eps)])
                f.write(line)
                print("Detection Result: " + str(alarm_value))

            if not self.use_saved_anomaly_score:
                filename = self.anomaly_score_filename.replace("*", str(self.game_id))

                with open(filename, "wb") as f:
                    np.save(f, np.asarray(self.anomaly_arr))

                reward_filename = self.reward_filename.replace("*", str(self.game_id))
                with open(reward_filename, "wb") as f:
                    np.save(f, np.asarray(reward))

                action_change_filename = self.action_change_filename.replace("*", str(self.game_id))
                with open(action_change_filename, "wb") as f:
                    np.save(f, np.asarray(self.action_change_arr))

                alarm_filename = self.anomaly_alarm_filename.replace("*", str(self.game_id))
                with open(alarm_filename, "wb") as f:
                    np.save(f, np.asarray(self.anomaly_alarm_arr))

        #print(self.anomaly_arr)
        #print(self.anomaly_alarm_arr)
        self.clean()

    def clean(self):
        self.action_change_arr.clear()
        self.anomaly_arr.clear()
        self.actions_arr.clear()
        self.anomaly_alarm_arr.clear()
        self.detection_queue.clear()
        self.test.setup_test(self.train)

    def check_use_saved_anomaly_score(self, num_games):
        anomaly_score_filenames = glob.glob(self.anomaly_score_filename)

        max_file = self.anomaly_score_filename.replace("*", str(num_games - 1))

        if max_file in anomaly_score_filenames:
            self.use_saved_anomaly_score = True
        else:
            print("Cannot use saved anomaly scores. No record of prior games.")
            self.use_saved_anomaly_score = False

        print("Reuse saved anomaly score: {}".format(self.use_saved_anomaly_score))

        return self.use_saved_anomaly_score

    def load_saved_anomaly_score(self, game_id):
        anomaly_score_filename = self.anomaly_score_filename.replace("*", str(game_id))
        reward_filename = self.reward_filename.replace("*", str(game_id))

        print("load anomaly scores from file: " + anomaly_score_filename)

        with open(anomaly_score_filename, "rb") as f:
            anomaly_scores = np.load(f)

        with open(reward_filename, "rb") as f:
            reward = np.load(f).item()
            
        return anomaly_scores, reward

    def process_step(self, y, frame_idx, game_id, action_change_val,
                     action_distribution=None, predicted_action=None, predicted_dist=None):
        self.game_id = game_id
        score = 1
        alarm = 0

        if self.technique == "KL":
            if self.use_saved_anomaly_score:
                score = y
                alarm = self.detection_process(score, frame_idx)

            else:
                self.test.update_action(y)
                score = self.test.compare(self.train, self.test)
                alarm = self.detection_process(score, frame_idx)

        elif self.technique == "VF" and predicted_dist is not None:
            score = abs(scipy.special.softmax(action_distribution) - scipy.special.softmax(predicted_dist)).sum() / \
                   len(action_distribution)
            if score > self.threshold:

                alarm = 1
            else:
                alarm = 0

        self.action_change_arr.append(action_change_val)
        self.anomaly_arr.append(score)
        self.anomaly_alarm_arr.append(alarm)
        self.actions_arr.append(y)

        return score, alarm

    def is_alarmed(self):
        return np.max(self.anomaly_alarm_arr)

    def detection_process(self, score, frame_idx):

        if score > self.threshold:
            alarm = 1
        else:
            alarm = 0

        if frame_idx <= self.skipped_frames:
            return 0

        if self.technique == "KL":
            self.detection_queue.append(alarm)

            if len(self.detection_queue) > self.queue_size:
                # check if too many steps exceed this threshold
                avg = np.mean(np.asarray(self.detection_queue))
                self.detection_queue.popleft()
                if avg >= self.alarm_percentage:
                    alarm = 1
                else:
                    alarm = 0
            else:
                return 0

        return alarm







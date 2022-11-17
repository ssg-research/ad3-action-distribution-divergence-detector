
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
import pandas as pd
import itertools


ENV_NAME = "Freeway"
ADVERSARIES = ["none", "fgsm", "uap_s", "uap_f", "obs_fgsm_wb", "obs_fgsm_wb_ingame"]
ATTACK_ONLY_ADV = ["fgsm", "uap_s", "uap_f", "obs_fgsm_wb", "obs_fgsm_wb_ingame"]

NO_ADV = "none"

CSV_FILENAME = "../anomaly_result/{}detection_result.csv".format(ENV_NAME)

COLUMN_NAMES = ["env", "agent", "adversary", "games" ,"skipped_frames", "queue_size",
                "percentile", "alarm_percentage", "reward", "detection", "eps"]

AGENTS = ["dqn", 'a2c', 'ppo']
TOTAL_GAMES = 10

def parse_key(key_str):
    game, frames, percentile, alar_perc, queue_size = key_str.split("-")

    return int(game), int(frames), int(percentile), float(alar_perc), int(queue_size)

df = pd.read_csv(CSV_FILENAME, names=COLUMN_NAMES)


# convert all the parameter combinations in the data into key strings
parameter_combos = set()
agent = AGENTS[0]
adv = NO_ADV
temp_df = df.loc[(df['agent'] == agent) &(df['adversary'] == adv)]

print("Load data...")
for index, row in temp_df.iterrows():
    key_str = "-".join([str(row['games']), str(row['skipped_frames']), str(row['percentile']), str(row['alarm_percentage']), str(row['queue_size'])])

    parameter_combos.add(key_str)

parameter_combos = list(parameter_combos)

sub_divide_data = dict()

print("Process data...")
for agent, adv in itertools.product(AGENTS, ADVERSARIES):
    if not agent in sub_divide_data:
        sub_divide_data[agent] = dict()

    if not adv in sub_divide_data[agent]:
        sub_divide_data[agent][adv] = dict()

    for key_str in parameter_combos:
        games, frames, percentile, alar_perc, queue_size = parse_key(key_str)
        temp = df.loc[(df['games'] == games) & (df['agent'] == agent) & (df['adversary'] == adv) & (
                    df['skipped_frames'] == frames) & (df['queue_size'] == queue_size) & (
                                  df['percentile'] == percentile) & (df['alarm_percentage'] == alar_perc)]

        sub_divide_data[agent][adv][key_str] = temp


print("Calculate different metrics...")
false_positive_rates = dict()
for agent in AGENTS:
    false_positive_rates[agent] = dict()

    for key_str in parameter_combos:
        temp_arr = sub_divide_data[agent][NO_ADV][key_str]["detection"].to_numpy()
        false_positive_rates[agent][key_str] = np.mean(temp_arr)

true_positive_rates = dict()
true_pos_rates_by_adv = dict()

for agent in AGENTS:
    true_positive_rates[agent] = dict()
    true_pos_rates_by_adv[agent] = dict()
    for key_str in parameter_combos:

        temp_arr = []

        for adv in ATTACK_ONLY_ADV:
            ar = sub_divide_data[agent][adv][key_str]["detection"].to_numpy()
            temp_arr = np.concatenate((temp_arr, ar))

            if adv not in true_pos_rates_by_adv[agent]:
                true_pos_rates_by_adv[agent][adv] = dict()
            true_pos_rates_by_adv[agent][adv][key_str] = np.mean(ar)

        true_positive_rates[agent][key_str] = np.mean(temp_arr)

balanced_accuracy = dict()

for agent in AGENTS:
    balanced_accuracy[agent] = dict()

    for key_str in parameter_combos:
        fpr = false_positive_rates[agent][key_str]
        tpr = true_positive_rates[agent][key_str]

        fnr = 1 - tpr
        tnr = 1 - fpr

        bal_acc = (tpr + tnr) / 2

        balanced_accuracy[agent][key_str] = bal_acc

true_positive = dict()
false_positive = dict()
true_negative = dict()
false_negative = dict()

for agent in AGENTS:
    true_positive[agent] = dict()
    false_positive[agent] = dict()
    true_negative[agent] = dict()
    false_negative[agent] = dict()

    for key_str in parameter_combos:

        normal_arr = sub_divide_data[agent][NO_ADV][key_str]["detection"].to_numpy()

        adv_arr = []

        for adv in ATTACK_ONLY_ADV:
            adv_arr = np.concatenate((adv_arr, sub_divide_data[agent][adv][key_str]["detection"].to_numpy()))

        tp = np.sum(adv_arr)
        fp = np.sum(normal_arr)
        tn = len(normal_arr) - fp
        fn = len(adv_arr) - tp

        true_positive[agent][key_str] = tp
        false_positive[agent][key_str] = fp
        true_negative[agent][key_str] = tn
        false_negative[agent][key_str] = fn

true_positive_adv = dict()
false_positive_adv = dict()
true_negative_adv = dict()
false_negative_adv = dict()
percision_adv = dict()
recall_adv = dict()
f1_score_adv = dict()

for agent in AGENTS:
    true_positive_adv[agent] = dict()
    false_positive_adv[agent] = dict()
    true_negative_adv[agent] = dict()
    false_negative_adv[agent] = dict()
    percision_adv[agent] = dict()
    recall_adv[agent] = dict()
    f1_score_adv[agent] = dict()

    for key_str in parameter_combos:

        normal_arr = sub_divide_data[agent][NO_ADV][key_str]["detection"].to_numpy()

        for adv in ATTACK_ONLY_ADV:

            if adv not in true_positive_adv[agent]:
                true_positive_adv[agent][adv] = dict()
                false_positive_adv[agent][adv] = dict()
                true_negative_adv[agent][adv] = dict()
                false_negative_adv[agent][adv] = dict()
                percision_adv[agent][adv] = dict()
                recall_adv[agent][adv] = dict()
                f1_score_adv[agent][adv] = dict()

            adv_arr = sub_divide_data[agent][adv][key_str]["detection"].to_numpy()

            tp = np.sum(adv_arr)
            fp = np.sum(normal_arr)
            tn = len(normal_arr) - fp
            fn = len(adv_arr) - tp

            true_positive_adv[agent][adv][key_str] = tp
            false_positive_adv[agent][adv][key_str] = fp
            true_negative_adv[agent][adv][key_str] = tn
            false_negative_adv[agent][adv][key_str] = fn

            percision_adv[agent][adv][key_str] = tp / (tp + fp)
            recall_adv[agent][adv][key_str] = tp / (tp + fn)

            f1_score_adv[agent][adv][key_str] = tp / (tp + 0.5 * (fp + fn))

percision = dict()
recall = dict()
f1_score = dict()

for agent in AGENTS:
    percision[agent] = dict()
    recall[agent] = dict()
    f1_score[agent] = dict()

    for key_str in parameter_combos:
        tp = true_positive[agent][key_str]
        fp = false_positive[agent][key_str]
        tn = true_negative[agent][key_str]
        fn = false_negative[agent][key_str]

        percision[agent][key_str] = tp / (tp + fp)
        recall[agent][key_str] = tp / (tp + fn)

        f1_score[agent][key_str] = tp / (tp + 0.5 * (fp + fn))
max_accuracy = dict()
max_recall = dict()
max_percision = dict()
max_tpr = dict()
max_f1_score = dict()

for agent in AGENTS:

    max_tup = ("", -1)

    for key, val in balanced_accuracy[agent].items():

        if val > max_tup[1]:
            max_tup = (key, val)

    max_accuracy[agent] = max_tup

    max_tup = ("", -1)

    for key, val in percision[agent].items():

        if val > max_tup[1]:
            max_tup = (key, val)

    max_percision[agent] = max_tup

    max_tup = ("", -1)

    for key, val in recall[agent].items():

        if val > max_tup[1]:
            max_tup = (key, val)

    max_recall[agent] = max_tup

    max_tup = ("", -1)

    for key, val in true_positive_rates[agent].items():

        if val > max_tup[1]:
            max_tup = (key, val)

    max_tpr[agent] = max_tup

    max_tup = ("", -1)

    for key, val in f1_score[agent].items():

        if val > max_tup[1]:
            max_tup = (key, val)

    max_f1_score[agent] = max_tup


print("Max accuracy: {}".format(max_accuracy))
print("Max percision: {}".format(max_percision))
print("Max recall: {}".format(max_recall))
print("Max TPR: {}".format(max_tpr))

print("Best argument for agents.....")
for agent in AGENTS:
    tup = max_f1_score[agent]
    print(tup)
    print("agent: " + agent)
    print("Accuracy: {}  Percision: {}   Recall: {} F1: {}".format(tup[1], percision[agent][tup[0]],
                                                            recall[agent][tup[0]], f1_score[agent][tup[0]]))
    train_games, skipped_frames, percentile, alar_perc, queue_size = tup[0].split("-")

    print("TPR: {}  FPR: {}".format(true_positive_rates[agent][tup[0]], false_positive_rates[agent][tup[0]]))
    print("Training games: {}".format(train_games))
    print("SKipped frames: {}".format(skipped_frames))
    print("Threshold percentile: {}".format(percentile))
    print("Alarm pertage: {}".format(alar_perc))
    print("Queue Size: {}".format(queue_size))

    print("---------------------")

for agent in AGENTS:
    for adv in ATTACK_ONLY_ADV:
        key, val = max_f1_score[agent]
        print(key)
        precision = percision_adv[agent][adv][key]
        recall = recall_adv[agent][adv][key]
        print("Agent: {}   adv: {}  Precision: {}   Recall: {} TPR: {}".format(agent, adv, precision, recall,
                                                                               true_pos_rates_by_adv[agent][adv][key]))











































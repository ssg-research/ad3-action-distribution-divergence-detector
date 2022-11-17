
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

import collections
import numpy as np
from scipy.stats import entropy

class KLDivergence:

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.distribution = np.zeros((self.num_actions, self.num_actions))
        self.action_bigram_count = np.zeros((self.num_actions, self.num_actions))
        self.Y = []

    def compare(self, a, b):
        n_div_flat = np.concatenate(a.distribution, axis=0) + 0.00001
        a_div_flat = np.concatenate(b.distribution, axis=0) + 0.00001
        e = entropy(n_div_flat, qk=a_div_flat)

        return e

    def update_action(self, y):
        self.Y.append(y)
        if len(self.Y) > 1:
            self.action_bigram_count[self.Y[-2]][self.Y[-1]] += 1
            self.distribution = self.action_bigram_count / (len(self.Y) - 1)

    def save_model_to_file(self, filename):
        np.savez(filename, distr=self.distribution, action_bigram=self.action_bigram_count)

    def load_model_from_file(self, filename):
        data = np.load(filename)
        self.distribution = data["distr"]
        self.action_bigram_count = data["action_bigram"]

    def load_distribution(self, distr):
        self.distribution = distr

    def get_distribution(self):
        return self.distribution

    def setup_test(self, trained):
        return

    def copy(self, div):
        self.num_actions = div.num_actions
        self.distribution = np.array(div.distribution, copy=True)
        self.action_bigram_count = np.array(div.action_bigram_count, copy=True)

    def clean(self):
        self.Y = []
        self.distribution = np.zeros((self.num_actions, self.num_actions))
        self.action_bigram_count = np.zeros((self.num_actions, self.num_actions))





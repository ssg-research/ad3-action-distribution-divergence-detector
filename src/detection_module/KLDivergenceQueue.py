
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


from detection_module.KLDivergence import KLDivergence


class KLDivergenceQueue(KLDivergence):

    def __init__(self, num_actions, queue_size):
        self.num_actions = num_actions
        self.distribution = np.zeros((self.num_actions, self.num_actions))
        self.action_bigram_count = np.zeros((self.num_actions, self.num_actions))

        self.queue_size = queue_size
        self.Y = collections.deque()

    def update_action(self, y):
        self.Y.append(y)
        if len(self.Y) > 1:
            self.action_bigram_count[self.Y[-2]][self.Y[-1]] += 1
            self.distribution = self.action_bigram_count / self.action_bigram_count.sum()

        if len(self.Y) >= self.queue_size:
            e1 = self.Y.popleft()
            e2 = self.Y.popleft()

            self.action_bigram_count[e1][e2] -= 1
            self.distribution = self.action_bigram_count / self.action_bigram_count.sum()

            self.Y.appendleft(e2)

    def setup_test(self, trained):
        self.copy(trained)
        self.Y.clear()

    def clean(self):
        self.Y = []
        self.distribution = np.zeros((self.num_actions, self.num_actions))
        self.action_bigram_count = np.zeros((self.num_actions, self.num_actions))





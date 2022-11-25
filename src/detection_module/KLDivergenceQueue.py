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





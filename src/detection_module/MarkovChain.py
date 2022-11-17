import itertools
import numpy as np
from collections import deque



class MarkovChain:

    # careful with the number of order, because the order will blow up
    def __init__(self, num_states, states=None, order=1):
        self.order = order
        self.num_states = num_states
        self.total_samples = 0

        self.past_actions = deque(maxlen=self.order)

        if not states:
            self.states = np.arange(num_states)
        else:
            self.states = states

        self.prior_states = np.flip(np.array(list(itertools.product(self.states, repeat=self.order))), axis=1)

        self.transition_matrix = np.zeros(shape=(self.num_states, self.num_states ** self.order))
        self.samples_matrix = np.zeros(shape=(self.num_states, self.num_states ** self.order))

        self.past_scores = []

    def load_distribution(self, distr):
        self.transition_matrix = distr

    def get_distribution(self):
        return self.transition_matrix

    def prior_states_idx(self, states):
        idx = 0
        for i in range(self.order):
            idx += states[i] * (self.num_states ** i)

        return idx

    def clear(self):
        self.transition_matrix = np.zeros(shape=(self.num_states, self.num_states ** self.order))
        self.samples_matrix = np.zeros(shape=(self.num_states, self.num_states ** self.order))
        self.past_scores = []

    def clear_past_action(self):
        self.past_actions.clear()
        self.past_scores = []

    def update_sample_matrix(self, prior_states, cur_state):
        idx = self.prior_states_idx(prior_states)
        self.samples_matrix[cur_state][idx] += 1
        self.total_samples += 1
        self.transition_matrix = self.samples_matrix / self.total_samples

    def update_action(self, y):
        if len(self.past_actions) == self.order:
            self.update_sample_matrix(list(self.past_actions), y)
            self.past_actions.pop()

        self.past_actions.appendleft(y)

    def detection(self, y):
        score = 1
        if len(self.past_actions) == self.order:
            idx = self.prior_states_idx(list(self.past_actions))
            score = self.transition_matrix[y][idx]
            self.past_actions.pop()

        self.past_actions.appendleft(y)
        self.past_scores.append(score)

        return np.mean(self.past_scores)



#!/usr/bin/env python

"""
  Author: Paniz Behboudian

"""
from __future__ import division
from rl_glue import BaseAgent

import numpy as np
import numpy.random as rnd


class Sarsa0(BaseAgent):

    def __init__(self, epsilon=None, number_of_actions=None, number_of_rows=None, number_of_columns=None, thau=1,
                 alpha=None, gamma=None):
        self.Q = None
        self.current_state = None
        self.last_action = None
        self.episode_num = None
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.path = []
        self.steps = 0
        self.thau = thau
        self.random_seed = None
        self.rng = None

    def agent_init(self):

        self.steps = 0
        self.path = []
        self.episode_num = 1
        assert self.number_of_rows > 0 and self.number_of_columns > 0, self.number_of_actions > 0
        assert self.alpha is not None and self.gamma is not None and self.random_seed is not None
        self.rng = np.random.RandomState(self.random_seed)
        self.Q = np.zeros((self.number_of_rows + 2, self.number_of_columns + 2, self.number_of_actions))

    def _choose_action(self, state):

        greedy = self.rng.choice(np.flatnonzero(self.Q[state[0], state[1]] == self.Q[state[0], state[1]].max()))
        prob = self.rng.uniform()
        if prob < self.epsilon:
            action = self.rng.randint(self.number_of_actions)
        else:
            action = greedy
        return action

    def _update_Q(self, state=None, action=None, reward=None, end=False):
        # Sarsa0
        if end:
            delta_Q = reward + self.gamma * 0 - self.Q[
                self.current_state[0], self.current_state[1], self.last_action]
            self.Q[self.current_state[0], self.current_state[1], self.last_action] = self.Q[self.current_state[0],
                                                                                            self.current_state[
                                                                                                1], self.last_action] + self.alpha * delta_Q
        else:
            delta_Q = reward + self.gamma * (self.Q[state[0], state[1], action]) - self.Q[
                self.current_state[0], self.current_state[1], self.last_action]
            self.Q[self.current_state[0], self.current_state[1], self.last_action] = self.Q[self.current_state[0],
                                                                                            self.current_state[
                                                                                                1], self.last_action] + self.alpha * delta_Q
        self.alpha *= self.thau

    def agent_start(self, state):
        """
        Arguments: state: numpy array
        Returns: action: integer
        """
        self.path = []
        self.path.append(state)
        action = self._choose_action(state=state)
        self.current_state = np.asarray(state)
        self.last_action = action
        return action

    def agent_step(self, reward, state):
        """
        Arguments: reward: floting point, state: integer
        Returns: action: floating point
        """

        self.steps += 1
        self.path.append(state)

        action = self._choose_action(state=state)
        self._update_Q(state=state, action=action, reward=reward, end=False)
        self.current_state = state
        self.last_action = action
        return action

    def agent_end(self, reward):
        """
        Arguments: reward: floating point
        Returns: Nothing
        """
        self.steps += 1
        self.path.append([-1, -1])
        self._update_Q(reward=reward, end=True)
        return

    def agent_message(self, in_message):
        if in_message.split(" ")[0] == 'alpha':
            self.alpha = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'gamma':
            self.gamma = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'thau':
            self.thau = float(in_message.split(" ")[1])
        else:
            raise Exception('Invalid AGENT message')

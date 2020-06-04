#!/usr/bin/env python

"""
  Author: Paniz Behboudian

"""
from __future__ import division
from rl_glue import BaseAgent

import numpy as np
import numpy.random as rnd
import math


class PiesAgent(BaseAgent):
    def __init__(self, epsilon=None, number_of_actions=None, number_of_rows=None, number_of_columns=None,
                 advice=None, decay=None, gamma=None, goal_coord=None, decay_param=None,
                 alpha=None, beta=None, thau=None, c=1, gamma_phi=None):
        self.Q = None
        self.Phi = None
        self.xi = None
        self.current_state = None
        self.last_action = None
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if gamma_phi is None:
            self.gamma_phi = gamma
        else:
            self.gamma_phi = gamma_phi
        self.path = []
        self.steps = 0
        self.thau = thau
        self.random_seed = None
        self.advice_scheme = advice
        self.rng = None
        self.goal_coord = goal_coord
        self.decay = decay
        self.decay_param = decay_param
        if self.advice_scheme == 'c_advice':
            self.c = c

    def agent_init(self):

        self.steps = 0
        self.path = []
        self.xi = 1
        assert self.number_of_rows > 0 and self.number_of_columns > 0, self.number_of_actions > 0
        assert self.alpha is not None and self.gamma is not None and self.random_seed is not None
        self.rng = np.random.RandomState(self.random_seed)
        self.Q = np.zeros((self.number_of_rows + 2, self.number_of_columns + 2, self.number_of_actions))
        self.Phi = np.zeros((self.number_of_rows + 2, self.number_of_columns + 2, self.number_of_actions))

    def _choose_action(self, state, policy='shaped'):
        if policy == 'Q':
            greedy = self.rng.choice(np.flatnonzero(
                self.Q[state[0], state[1]] == (self.Q[state[0], state[1]]).max()))
        elif policy == 'shaped':
            greedy = self.rng.choice(np.flatnonzero(
                self.Q[state[0], state[1]] - self.xi * self.Phi[state[0], state[1]] == (
                        self.Q[state[0], state[1]] - self.xi * self.Phi[state[0], state[1]]).max()))
        elif policy == 'Phi':
            greedy = self.rng.choice(np.flatnonzero(
                self.Phi[state[0], state[1]] == (self.Phi[state[0], state[1]]).max()))
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
            delta_Q = reward + self.gamma * self.Q[state[0], state[1], action] - self.Q[
                self.current_state[0], self.current_state[1], self.last_action]
            self.Q[self.current_state[0], self.current_state[1], self.last_action] = self.Q[self.current_state[0],
                                                                                            self.current_state[
                                                                                                1], self.last_action] + self.alpha * delta_Q
        self.alpha *= self.thau

    def _update_Phi(self, state=None, action=None, reward=None, end=False):
        if end:
            delta_Phi = reward + self.gamma_phi * 0 - self.Phi[
                self.current_state[0], self.current_state[1], self.last_action]
            self.Phi[self.current_state[0], self.current_state[1], self.last_action] = self.Phi[self.current_state[0],
                                                                                                self.current_state[
                                                                                                    1], self.last_action] + self.beta * delta_Phi
        else:
            delta_Phi = reward + self.gamma_phi * self.Phi[state[0], state[1], action] - self.Phi[
                self.current_state[0], self.current_state[1], self.last_action]
            self.Phi[self.current_state[0], self.current_state[1], self.last_action] = self.Phi[self.current_state[0],
                                                                                                self.current_state[
                                                                                                    1], self.last_action] + self.beta * delta_Phi

    def agent_start(self, state):
        """
        Arguments: state: numpy array
        Returns: action: integer
        """
        self.path = []
        self.path.append(state)
        action = self._choose_action(state=state, policy='shaped')
        self.current_state = np.asarray(state)
        self.last_action = action
        return action

    def agent_step(self, reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
        """
        Arguments: reward: floting point, state: integer
        Returns: action: floating point
        """

        self.steps += 1
        self.path.append(state)
        action = self._choose_action(state=state, policy='shaped')
        if self.xi > 0:
            reward_phi = -self.intrinsic_reward(sp=state)
            self._update_Phi(state=state, action=self._choose_action(state=state, policy='Phi'), reward=reward_phi,
                             end=False)
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
        if self.xi > 0:
            reward_phi = -self.intrinsic_reward(sp=self.goal_coord)
            self._update_Phi(reward=reward_phi, end=True)
        self._update_Q(reward=reward, end=True)
        return

    def intrinsic_reward(self, sp):
        """
        :param sp: next state
        :return: expert advice for the state transition
        """
        i_reward = 0
        if 'defined' in self.advice_scheme:
            reward_transitions = None
            if 'good' in self.advice_scheme:
                reward_transitions = {(1, 1): (1, 2), (2, 1): (1, 1), (2, 2): (2, 1)}
            elif 'bad' in self.advice_scheme:
                reward_transitions = {(1, 1): (2, 1), (2, 1): (2, 2), (2, 2): (2, 1)}
            else:
                raise Exception('Invalid defined advice')
            for s in reward_transitions.keys():
                if s[0] == self.current_state[0] and s[1] == self.current_state[1] \
                        and reward_transitions[s][0] == sp[0] and reward_transitions[s][1] == sp[1]:
                    i_reward = 1
        elif 'c_advice' in self.advice_scheme:
            # right and down
            if self.last_action == 1 or self.last_action == 2:
                i_reward = 1
        else:
            raise Exception('Invalid Advice Scheme')
        return i_reward

    def agent_message(self, in_message):
        if in_message.split(" ")[0] == 'alpha':
            self.alpha = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'gamma':
            self.gamma = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'beta':
            self.beta = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'thau':
            self.thau = float(in_message.split(" ")[1])
        elif in_message.split(" ")[0] == 'decay_param':
            self.decay_param = float(in_message.split(" ")[1])
        else:
            raise Exception('Invalid AGENT message')

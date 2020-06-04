#!/usr/bin/env python

"""
  Author: Paniz Behboudian

"""
from __future__ import division
from rl_glue import RLGlue
from dpba_agent import DPBAAgent
from pies_agent import PiesAgent
from sarsa0_agent import Sarsa0
from gridworld_env import GridWorldEnv
from toy_env import ToyEnv
import numpy as np
import os
import configparser
import itertools


def compute_epsilon(current_epsilon=0):
    return current_epsilon - (
            (float(initial_params['initial_epsilon']) - float(initial_params['final_epsilon'])) / int(
        initial_params['total_episodes']))


def compute_xi(current_xi=0, decay=None, decay_param=None):
    if decay == 'Linear':
        if (current_xi - 1 / decay_param) < 0:
            return 0
        else:
            return current_xi - 1 / decay_param
    else:
        raise Exception('Invalid decay!')


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('experimental_config.ini')
    # list the section names from config file which you want to run
    section_names = ['sarsa0-gridworld']
    initial_params = {'env': '', 'agent': '', 'num_runs': '', 'initial_epsilon': '', 'final_epsilon': '',
                      'number_of_rows': '', 'number_of_columns': '', 'actions': '', 'policy': '', 'advice': '',
                      'intent': '', 'alpha': '', 'beta': '', 'gamma': '', 'decay': '', 'decay_param': '',
                      'total_episodes': 500, 'max_episode_steps': 10000, 'thau': 1, 'initial_phi': 0, 'gamma_phi': ''}
    file_name_params = {'agent', 'policy', 'alpha', 'beta', 'initial_epsilon', 'final_epsilon', 'advice',
                        'decay', 'decay_param', 'num_runs', 'total_episodes'}
    # Instead of using the initialized values from initial_params dictionary,
    # we can use grid search for parameter study and iterate over grid_params.
    # Example: grid_search_params = {'alpha':[0.2, 0.1]}
    grid_search_params = {}
    sorted_params_names = sorted(grid_search_params.keys())
    grid_size = 1
    for value in grid_search_params.values():
        grid_size *= len(value)
    test_coord = [[_ for _ in range(0, len(grid_search_params[param]))] for param in
                  sorted_params_names]

    for section in config.sections():
        if section in section_names:
            for initial_param in initial_params:
                if initial_param in config[section]:
                    initial_params[initial_param] = config[section][initial_param]
                elif initial_param in config['Default']:
                    initial_params[initial_param] = config['Default'][initial_param]
            print(initial_params)
            start_coord = np.zeros(2)
            goal_coord = np.zeros(2)
            plot_file_directory = 'plot_files/' + initial_params['env'] + "/" + initial_params['agent'] + "/"
            max_episode_steps = int(initial_params['max_episode_steps'])

            # data visualization parameters
            episodes_steps = np.zeros((grid_size, int(initial_params['num_runs']),
                                       int(initial_params['total_episodes']) + 10))
            Q_t = np.zeros(
                (grid_size, int(initial_params['num_runs']), int(initial_params['total_episodes']),
                 int(initial_params['number_of_rows']) + 2, int(initial_params['number_of_columns']) + 2,
                 int(initial_params['actions'])))
            Phi_t = np.zeros(
                (grid_size, int(initial_params['num_runs']), int(initial_params['total_episodes']),
                 int(initial_params['number_of_rows']) + 2, int(initial_params['number_of_columns']) + 2,
                 int(initial_params['actions'])))

            j = 0
            for combination in itertools.product(*test_coord):
                # 20x20 Grid-World
                if 'gridworld_env' in initial_params['env']:
                    env = GridWorldEnv(number_of_columns=int(initial_params['number_of_columns']),
                                       number_of_rows=int(initial_params['number_of_rows']))
                # Toy Example
                elif 'toy_env' in initial_params['env']:
                    env = ToyEnv(number_of_columns=int(initial_params['number_of_columns']),
                                 number_of_rows=int(initial_params['number_of_rows']))
                else:
                    raise Exception('Invalid env_name in exp')
                if 'dpba' in initial_params['agent']:
                    agent = DPBAAgent(epsilon=float(initial_params['initial_epsilon']),
                                      number_of_actions=int(initial_params['actions']),
                                      number_of_columns=int(initial_params['number_of_columns']),
                                      number_of_rows=int(initial_params['number_of_rows']),
                                      policy=initial_params['policy'],
                                      advice=initial_params['advice'],
                                      gamma=float(initial_params['gamma']),
                                      goal_coord=env.goal_coord, thau=float(initial_params['thau']),
                                      alpha=float(initial_params['alpha']),
                                      beta=float(initial_params['beta']),
                                      initial_phi=float(initial_params['initial_phi']))
                elif 'pies' in initial_params['agent']:
                    agent = PiesAgent(epsilon=float(initial_params['initial_epsilon']),
                                      number_of_actions=int(initial_params['actions']),
                                      number_of_columns=int(initial_params['number_of_columns']),
                                      number_of_rows=int(initial_params['number_of_rows']),
                                      advice=initial_params['advice'], decay=initial_params['decay'],
                                      decay_param=float(initial_params['decay_param']),
                                      gamma=float(initial_params['gamma']),
                                      goal_coord=env.goal_coord, thau=float(initial_params['thau']),
                                      alpha=float(initial_params['alpha']),
                                      beta=float(initial_params['beta']), c=1)
                elif 'sarsa0' in initial_params['agent']:
                    agent = Sarsa0(epsilon=float(initial_params['initial_epsilon']),
                                   number_of_actions=int(initial_params['actions']),
                                   number_of_columns=int(initial_params['number_of_columns']),
                                   number_of_rows=int(initial_params['number_of_rows']),
                                   thau=float(initial_params['thau']),
                                   alpha=float(initial_params['alpha']), gamma=float(initial_params['gamma']))
                else:
                    raise Exception('Invalid agent name in exp')
                i = 0
                for key in sorted_params_names:
                    msg_to_send = key + ' ' + str(grid_search_params[key][combination[i]])
                    print(msg_to_send)
                    agent.agent_message(msg_to_send)
                    i += 1
                rl_glue = RLGlue(agent_obj=agent, env_obj=env)
                for r in range(int(initial_params['num_runs'])):
                    print('run: ' + str(r))
                    agent.random_seed = r
                    rl_glue.rl_init()
                    agent.epsilon = float(initial_params['initial_epsilon'])
                    for e in range(int(initial_params['total_episodes'])):
                        rl_glue.rl_episode(max_episode_steps)
                        agent.epsilon = compute_epsilon(current_epsilon=agent.epsilon)
                        episodes_steps[j, r, e] = rl_glue.num_ep_steps()
                        Q_t[j, r, e] = agent.Q
                        if initial_params['agent'] != 'sarsa0':
                            Phi_t[j, r, e] = agent.Phi
                        if 'pies' in initial_params['agent']:
                            agent.xi = compute_xi(current_xi=agent.xi, decay=agent.decay,
                                                  decay_param=agent.decay_param)
                    print('path length', len(agent.path))
                j += 1

            # finding the best parameter setting for grid search based on AUC
            best_param_set_index = np.random.choice(
                np.flatnonzero(
                    np.trapz(np.mean(episodes_steps, 1)) == np.trapz(np.mean(episodes_steps, 1)).min()))
            best_params_string = ''
            for index in range(0, len(tuple(itertools.product(*test_coord))[best_param_set_index])):
                key_at_index = sorted_params_names[index]
                best_value_index_at_index = tuple(itertools.product(*test_coord))[best_param_set_index][index]
                print('best value for ', key_at_index)
                print(grid_search_params[key_at_index][best_value_index_at_index])
                best_params_string += ',' + key_at_index + '=' + str(
                    grid_search_params[key_at_index][best_value_index_at_index])

            # generating the file name and write the visualisation data
            for param in file_name_params:
                best_params_string += ',' + param + '=' + initial_params[param]
            os.makedirs(os.path.dirname(plot_file_directory), exist_ok=True)
            diff1 = float(np.trapz(np.mean(episodes_steps[best_param_set_index], axis=0)))
            np.save(plot_file_directory + "StepsPerEpisode," + str(diff1) + best_params_string,
                    np.mean(episodes_steps[best_param_set_index], axis=0))
            np.save(plot_file_directory + "STD-StepsPerEpisode," + str(diff1) + best_params_string,
                    np.std(episodes_steps[best_param_set_index], axis=0))
            np.save(plot_file_directory + 'Qt,' + str(diff1) + best_params_string,
                    np.mean(Q_t[best_param_set_index], axis=0))
            np.save(plot_file_directory + 'STD-Qt,' + str(diff1) + best_params_string,
                    np.std(Q_t[best_param_set_index], axis=0))
            np.savetxt(plot_file_directory + 'final_path', agent.path)
            np.save(plot_file_directory + 'Phit,' + str(diff1) + best_params_string,
                    np.mean(Phi_t[best_param_set_index], axis=0))
            np.save(plot_file_directory + 'STD-Phit,' + str(diff1) + best_params_string,
                    np.std(Phi_t[best_param_set_index], axis=0))

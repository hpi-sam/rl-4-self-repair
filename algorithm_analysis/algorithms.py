import itertools
import random
import time

import matplotlib.pyplot as plt

import gym
import numpy as np

from tqdm.auto import tqdm

from algorithm_analysis import benchmark
from algorithm_analysis.metrics import TabularMetric
from algorithm_analysis.plotting import plot_episode_length_over_time_tabular, plot_episode_reward_over_time_tabular
from envs.broken_components import BrokenComponentsEnv
from envs.data_handler import DataHandler


# epsilon greedy strategy to choose next state
# i.e. choose whether to exploit or explore the env
# epsilon == explore probability
def epsilon_greedy(Q, epsilon, state):
    """
    @param Q Q-table
    @param epsilon (exploration rate)
    @param state
    """
    # contains q_values for the state
    q_slice = Q[state, :]
    if np.random.rand() < epsilon:
        action = np.random.randint(0, len(q_slice))
    else:
        action = np.argmax(q_slice)

    return action


def run_single_estimator(alg, env, num_states, num_actions, episodes=1000,
                         min_explore_rate=0.1, max_explore_rate=1,
                         learning_rate=0.2, discount_rate=0.90, trace_decay=0.9):
    """

    :param alg: Choose from 'qlearning' and 'sarsa'
    :param min_explore_rate: Choose value > 0
    """

    # init Q-table
    Q = np.zeros((num_states, num_actions), dtype='float32')
    print(f'Run {alg} with {num_states} states and {num_actions} actions.')

    # calculate explore rate decay to hit the min_explore_rate at the last run.
    explore_rate_decay = (np.log(max_explore_rate - min_explore_rate) - np.log(min_explore_rate)) / episodes

    # init episode metrics
    explore_rate = max_explore_rate
    episode_explore_rates = []
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(episodes)):
        E = np.zeros((num_states, num_actions), dtype='float32')
        state = env.reset()
        action = epsilon_greedy(Q, explore_rate, state)
        total_reward = 0

        time_for_algo_steps = []
        time_for_partial_steps = []
        for step in itertools.count(1):
            t0 = time.monotonic()
            if alg == 'qlearning':
                action = epsilon_greedy(Q, explore_rate, state)

            next_state, reward, done, info = env.step(action)
            # print(f'Step: {step}, Action: {action}, Next State: {next_state}, Reward: {reward}')
            E[state, action] += 1

            if alg == 'qlearning':
                next_action = epsilon_greedy(Q, 0, next_state)
            elif alg == 'sarsa':
                next_action = epsilon_greedy(Q, explore_rate, next_state)
            else:
                raise NotImplementedError(alg)

            delta = reward + (discount_rate * Q[next_state, next_action]) - Q[state, action]

            t1 = time.monotonic()
            # update Q-Table for Q(s,a)
            Q += learning_rate * delta * E
            E = E * trace_decay * discount_rate
            time_for_partial_steps.append(time.monotonic()-t1)

            state = next_state
            action = next_action
            total_reward += reward

            time_for_algo_steps.append(time.monotonic()-t0)
            if done:
                break

        # total_time_in_algo = sum(time_for_algo_steps)
        # total_time_in_partial = sum(time_for_partial_steps)
        # print(f'Time spent in algo episode {total_time_in_algo}')
        # print(f'Time spent in partial episode {total_time_in_partial}')
        # print(f'Steps taken {step}')
        # update episode metrics
        episode_explore_rates.append(explore_rate)
        episode_lengths.append(step)
        episode_rewards.append(total_reward)

        # model exploration rate decay.
        explore_rate = (max_explore_rate - min_explore_rate) * np.exp(-explore_rate_decay * episode)
        if episode / episodes > 0.9:
            explore_rate = 0


    env.close()

    metric = TabularMetric(episodes, episode_lengths, episode_rewards, episode_explore_rates, learning_rate, discount_rate, trace_decay)
    return metric


def run_double_estimator(alg, env, num_states, num_actions, episodes=1000,
                         min_explore_rate=0.05, max_explore_rate=1,
                         learning_rate=0.2, discount_rate=0.90, trace_decay=0.9):
    """

    :param alg: Choose from 'qlearning' and 'sarsa'
    :param min_explore_rate: Choose value > 0
    """

    print(f'Run {alg} with {num_states} states and {num_actions} actions.')

    # calculate explore rate decay to hit the min_explore_rate at the last run.
    explore_rate_decay = (np.log(max_explore_rate - min_explore_rate) - np.log(min_explore_rate)) / episodes

    # init Q-table
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))

    # init episode metrics
    explore_rate = max_explore_rate
    episode_explore_rates = []
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(episodes)):
        E = np.zeros((num_states, num_actions))

        update_Q1 = np.random.rand() < 0.5
        Q = Q1 if update_Q1 else Q2

        state = env.reset()
        action = epsilon_greedy(Q, explore_rate, state)
        total_reward = 0

        for step in itertools.count(1):
            if alg == 'qlearning':
                action = epsilon_greedy(Q, explore_rate, state)

            E[state, action] += 1

            next_state, reward, done, info = env.step(action)

            if alg == 'qlearning':
                next_action = epsilon_greedy(Q, 0, next_state)
            elif alg == 'sarsa':
                next_action = epsilon_greedy(Q, explore_rate, next_state)
            else:
                raise NotImplementedError(alg)

            Q_max_value = Q2[next_state, next_action] if update_Q1 else Q1[next_state, next_action]
            delta = reward + discount_rate * Q_max_value - Q[state, action]

            # update Q-Table for Q(s,a)
            Q += learning_rate * delta * E
            E = E * trace_decay

            state = next_state
            action = next_action
            update_Q1 = np.random.rand() < 0.5
            Q = Q1 if update_Q1 else Q2

            total_reward += reward

            if done:
                break

        # update episode metrics
        episode_explore_rates.append(explore_rate)
        episode_lengths.append(step)
        episode_rewards.append(total_reward)

        # model exploration rate decay.
        explore_rate = (max_explore_rate - min_explore_rate) * np.exp(-explore_rate_decay * episode)
        if episode / episodes > 0.9:
            explore_rate = 0


    env.close()

    metrics = TabularMetric(episodes, episode_lengths, episode_rewards, episode_explore_rates, learning_rate, discount_rate, trace_decay)
    return metrics


if __name__ == '__main__':
    dataHandler = DataHandler()
    components_list = benchmark.BROKEN_COMPONENTS[4]
    env = BrokenComponentsEnv(components_list, reward_modus='raw')
    # env = gym.make('Taxi-v3')

    learning_rate = 0.1
    discount_rate = 0.9

    metric = run_single_estimator('sarsa', env, env.observation_space.n, env.action_space.n, episodes=300,
                                  learning_rate=learning_rate, discount_rate=discount_rate, trace_decay=0.9)

    for r in metric.rewards:
        print(r)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharex=True,
                            sharey='col')
    plot_episode_length_over_time_tabular(axs[0], metric, smoothing_window=5)
    plot_episode_reward_over_time_tabular(axs[1], metric, smoothing_window=5)
    plt.show()
    plt.plot(metric.explore_rates)
    plt.xlabel('Episode')
    plt.ylabel(f'Explore Rate')
    plt.title(f'Explore rate over time')
    plt.savefig(f'tabularplots/explorerates.png', dpi=300,
                bbox_inches='tight')
    plt.show()
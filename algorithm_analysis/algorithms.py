import itertools
import matplotlib.pyplot as plt

import gym
import numpy as np

from tqdm.auto import tqdm
from algorithm_analysis.metrics import Metric
from algorithm_analysis.plotting import plot_episode_length_over_time, plot_episode_reward_over_time
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


def run(alg, env, num_states, num_actions, episodes=1000,
        min_explore_rate=0.01, max_explore_rate=1, explore_rate_decay=0.005,
        learning_rate=0.2, discount_rate=0.90, trace_decay=0.9):
    """

    :param alg: Choose from 'qlearning' and 'sarsa'
    """

    # init Q-table
    Q = np.zeros((num_states, num_actions))
    print(f'Run {alg} with {num_states} states and {num_actions} actions.')

    # init episode metrics
    explore_rate = max_explore_rate
    episode_explore_rates = []
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(episodes)):
        E = np.zeros((num_states, num_actions))
        state = env.reset()
        action = epsilon_greedy(Q, explore_rate, state)
        total_reward = 0

        for step in itertools.count(1):
            E[state, action] += 1

            next_state, reward, done, info = env.step(action)

            if alg == 'qlearning':
                next_action = epsilon_greedy(Q, 0, next_state)
            elif alg == 'sarsa':
                next_action = epsilon_greedy(Q, explore_rate, next_state)
            else:
                raise NotImplementedError(alg)

            delta = reward + (discount_rate * Q[next_state, next_action]) - Q[state, action]

            # update Q-Table for Q(s,a)
            Q += learning_rate * delta * E
            E = E * trace_decay * discount_rate
            # else:
            #     Q[state, action] += learning_rate * (reward - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        # update episode metrics
        episode_explore_rates.append(explore_rate)
        episode_lengths.append(step)
        episode_rewards.append(total_reward)

        # model exploration rate decay.
        explore_rate = min_explore_rate + \
                       (max_explore_rate - min_explore_rate) * np.exp(-explore_rate_decay * episode)

    env.close()

    metrics = Metric(episodes, episode_lengths, episode_rewards, episode_explore_rates, learning_rate, discount_rate, trace_decay)
    return metrics


if __name__ == '__main__':
    dataHandler = DataHandler()
    broken_components = dataHandler.get_sample_component_failure_pairs(10)
    env = BrokenComponentsEnv(broken_components, reward_modus='raw')
    # env = gym.make('Taxi-v3')

    learning_rate = 0.01
    discount_rate = 0.9

    metric = run('qlearning', env, env.observation_space.n, env.action_space.n, episodes=200,
                 learning_rate=learning_rate, discount_rate=discount_rate, trace_decay=0.9)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharex=True,
                            sharey='col')
    plot_episode_length_over_time(axs[0], metric, smoothing_window=5)
    plot_episode_reward_over_time(axs[1], metric, smoothing_window=5)

    plt.show()

import random
from dataclasses import dataclass
from typing import List

import numpy as np

from envs.broken_components import BrokenComponentsEnv
from envs.data_handler import DataHandler


def run(env, num_states, num_actions, episodes=1000,
        min_explore_rate=0.01, max_explore_rate=1, explore_rate_decay=0.005,
        learning_rate=0.1, discount_rate=0.99):
    q_table = np.zeros((num_states, num_actions))
    print(f'Run q-learning with {num_states} states and {num_actions} actions.')

    # Initialize metrics
    metrics = Metric(episodes, [], [], [], learning_rate, discount_rate)
    explore_rate = max_explore_rate

    for episode in range(episodes):

        state = env.reset()
        episode_length = 0
        total_reward = 0
        metrics.explore_rates.append(explore_rate)

        while True:
            # Explore or exploit the env.
            if random.uniform(0, 1) < explore_rate:
                action = env.action_space.sample()
                print(f'Random action: {action}')
            else:
                action = np.argmax(q_table[state, :])
                print(f'State: {state}')
                print(f'Length of action_space for state {state} = : {q_table[state, :]}')
                print(f'optimal action: {action}')

            new_state, reward, done, info = env.step(action)

            # Update Q-Table for Q(s,a)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                     learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            total_reward += reward
            episode_length += 1

            if done:
                break

        # Model exploration rate decay.
        explore_rate = min_explore_rate + \
                       (max_explore_rate - min_explore_rate) * np.exp(-explore_rate_decay * episode)

        metrics.episode_lengths.append(episode_length)
        metrics.rewards.append(total_reward)

    env.close()
    return metrics


@dataclass
class Metric:
    episodes: int
    episode_lengths: List[int]
    rewards: List[np.float64]
    explore_rates: List[np.float64]
    learning_rate: np.float64
    discount_rate: np.float64


if __name__ == '__main__':
    dataHandler = DataHandler()
    broken_components = dataHandler.get_sample_component_failure_pairs(3)
    env = BrokenComponentsEnv(broken_components, reward_modus='raw')

    learning_rate = 0.06
    discount_rate = 0.06

    metric = run(env, env.observation_space.n, env.action_space.n, episodes=1000,
                           learning_rate=learning_rate, discount_rate=discount_rate)


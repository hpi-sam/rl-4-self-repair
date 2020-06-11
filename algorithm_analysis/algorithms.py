import gym
import numpy as np
import tqdm as tqdm

from tqdm.autonotebook import tqdm
from algorithm_analysis.metrics import Metric
from envs.broken_components import BrokenComponentsEnv
from envs.data_handler import DataHandler


# epsilon greedy strategy to choose next state
# i.e. choose whether to exploit or explore the env
def epsilon_greedy(Q, epsilon, state):
    """
    @param Q Q-table
    @param epsilon (exploration rate)
    @param state
    """
    # contains q_values for the state
    q_slice = Q[state, :]
    if np.random.rand() < epsilon:
        action = np.argmax(q_slice)
    else:
        action = np.random.randint(0, len(q_slice))
    return action


def run(alg, env, num_states, num_actions, episodes=1000,
        min_explore_rate=0.01, max_explore_rate=1, explore_rate_decay=0.005,
        learning_rate=0.2, discount_rate=0.90):
    """

    :param alg: Choose from 'qlearning' and 'sarsa'
    """

    # init Q-table
    Q = np.zeros((num_states, num_actions))
    print(f'Run q-learning with {num_states} states and {num_actions} actions.')

    # init episode metrics
    explore_rate = max_explore_rate
    episode_explore_rates = []
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(episodes)):

        s = env.reset()
        episode_length = 0
        total_reward = 0

        while True:
            a = epsilon_greedy(Q, explore_rate, s)

            s_next, reward, done, info = env.step(a)

            if alg == 'qlearning':
                a_next = epsilon_greedy(Q, 0, s)
            elif alg == 'sarsa':
                a_next = epsilon_greedy(Q, explore_rate, s)
            else:
                raise NotImplementedError(alg)

            # update Q-Table for Q(s,a)
            if not done:
                Q[s, a] += learning_rate * (reward + (discount_rate * Q[s_next, a_next]) - Q[s, a])
            else:
                Q[s, a] += learning_rate * (reward - Q[s, a])

            s = s_next
            total_reward += reward
            episode_length += 1

            if done:
                break

        # update episode metrics
        episode_explore_rates.append(explore_rate)
        episode_lengths.append(episode_length)
        episode_rewards.append(total_reward)

        # model exploration rate decay.
        explore_rate = min_explore_rate + \
                       (max_explore_rate - min_explore_rate) * np.exp(-explore_rate_decay * episode)

    env.close()

    metrics = Metric(episodes, episode_lengths, episode_rewards, episode_explore_rates, learning_rate, discount_rate)
    return metrics


if __name__ == '__main__':
    # dataHandler = DataHandler()
    # broken_components = dataHandler.get_sample_component_failure_pairs(3)
    # env = BrokenComponentsEnv(broken_components, reward_modus='raw')
    env = gym.make('Taxi-v3')

    learning_rate = 0.06
    discount_rate = 0.06

    metric = run('qlearning', env, env.observation_space.n, env.action_space.n, episodes=1000,
                 learning_rate=learning_rate, discount_rate=discount_rate)

    print(metric)

import pandas as pd
from algorithm_analysis.metrics import A2C_MetricByEpisodes, A2C_MetricByBatch


def plot_episode_length_over_time_tabular(ax, metric, smoothing_window=10):
    # Plot the episode length over time
    lengths_smoothed = pd.Series(metric.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(lengths_smoothed)
    ax.set_xlabel('Episode')
    # ax.set_ylabel('Episode Length')
    ax.set_title(f'Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')


def plot_episode_reward_over_time_tabular(ax, metric, smoothing_window=10):
    # Plot the episode reward over time
    rewards_smoothed = pd.Series(metric.rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel('Episode')
    # ax.set_ylabel(f'Episode Reward (Smoothed {smoothing_window})')
    ax.set_title(f'Reward over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')

    
def scatter_episode_length_over_time_tabular(ax, metric):
    # Plot the episode length over time
    #ax.scatter(range(metric.episodes), metric.episode_lengths, alpha=0.3)
    ax.scatter(range(metric.episodes), metric.episode_lengths, alpha=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title(f'Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')


def scatter_episode_reward_over_time_tabular(ax, metric):
    # Plot the episode reward over time
    ax.scatter(range(metric.episodes), metric.rewards, alpha=1)
    #ax.scatter(range(metric.episodes), metric.rewards, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Episode Reward')
    ax.set_title(f'Reward over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')


def plot_episode_length_over_time_A2C(ax, metric, smoothing_window=10):
    # Plot the episode length over time
    lengths_smoothed = pd.Series(metric.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(lengths_smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    
    if isinstance(metric, A2C_MetricByBatch):
        ax.set_title(f'{metric.num_broken_components}| Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} VC:{metric.value_coefficient} EC:{metric.entropy_coefficient} BS:{metric.batch_size} U:{metric.updates}')
    elif isinstance(metric, A2C_MetricByEpisodes):
        ax.set_title(f'{metric.num_broken_components}| Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} VC:{metric.value_coefficient} EC:{metric.entropy_coefficient}')


def plot_episode_reward_over_time_A2C(ax, metric, smoothing_window=10):
    # Plot the episode reward over time
    rewards_smoothed = pd.Series(metric.rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'{metric.num_broken_components}| Episode Reward (Smoothed {smoothing_window})')
    
    if isinstance(metric, A2C_MetricByBatch):
        ax.set_title(f'Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} VC:{metric.value_coefficient} EC:{metric.entropy_coefficient} BS:{metric.batch_size} U:{metric.updates}')
    elif isinstance(metric, A2C_MetricByEpisodes):
        ax.set_title(f'{metric.num_broken_components}| Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} VC:{metric.value_coefficient} EC:{metric.entropy_coefficient}')
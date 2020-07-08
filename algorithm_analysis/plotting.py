import pandas as pd


def plot_episode_length_over_time_tabular(ax, metric, smoothing_window=10):
    # Plot the episode length over time
    lengths_smoothed = pd.Series(metric.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(lengths_smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title(f'Length over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')


def plot_episode_reward_over_time_tabular(ax, metric, smoothing_window=10):
    # Plot the episode reward over time
    rewards_smoothed = pd.Series(metric.rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Episode Reward (Smoothed {smoothing_window})')
    ax.set_title(f'Reward over Time LR:{metric.learning_rate} DR:{metric.discount_rate} TD:{metric.trace_decay}')

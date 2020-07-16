import os
import pickle
from dataclasses import dataclass
from typing import List


def dump_metric(metric, dir_path, filename):
    with open(os.path.join(dir_path, filename), 'wb') as picklefile:
        pickle.dump(metric, picklefile)
        picklefile.close()

    
def load_metric(dir_path, filename):
    with open(os.path.join(dir_path, filename), 'rb') as picklefile:
        metric = pickle.load(picklefile)
        return metric


@dataclass(frozen=True)
class TabularMetric:
    episodes: int
    episode_lengths: List[int]
    rewards: List[float]
    explore_rates: List[float]
    learning_rate: float
    discount_rate: float
    trace_decay: float
        
        
@dataclass(frozen=True)
class A2C_MetricByBatch:
    batch_size: int
    updates: int
    episode_lengths: List[int]
    rewards: List[float]
    learning_rate: float
    discount_rate: float
    value_coefficient: float
    entropy_coefficient: float
    num_broken_components: int

@dataclass(frozen=True)        
class A2C_MetricByEpisodes:
    episodes: int
    episode_lengths: List[int]
    rewards: List[float]
    learning_rate: float
    discount_rate: float
    value_coefficient: float
    entropy_coefficient: float
    num_broken_components: int
    episode_skips: [int]

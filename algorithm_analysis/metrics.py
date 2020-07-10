from dataclasses import dataclass
from typing import List

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


@dataclass(frozen=True)        
class A2C_MetricByEpisodes:
    episodes: int
    episode_lengths: List[int]
    rewards: List[float]
    learning_rate: float
    discount_rate: float
    value_coefficient: float
    entropy_coefficient: float

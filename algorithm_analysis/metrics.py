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

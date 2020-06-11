from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Metric:
    episodes: int
    episode_lengths: List[int]
    rewards: List[float]
    explore_rates: List[float]
    learning_rate: float
    discount_rate: float

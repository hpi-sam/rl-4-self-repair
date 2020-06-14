import itertools
from typing import Tuple, List, Dict

import gym
import numpy as np
from gym import spaces

from envs.data_handler import DataHandler

DATA_HANDLER = DataHandler()


class BrokenComponentsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, broken_components: List[Tuple], reward_modus: str = 'raw', reward_decrease: bool = False, reward_decrease_factor: float = 0.99):
        super(BrokenComponentsEnv, self).__init__()
        self.data_handler = DATA_HANDLER
        self.reward_modus = reward_modus
        self.reward_decrease = reward_decrease
        self.reward_decrease_factor = reward_decrease_factor
        self.punishment = self.data_handler.data.max()*-1
        self.action_space, self.action_space_names = self.__create_action_space(broken_components)
        self.observation_space, self.observation_space_names = self.__create_observation_space(broken_components)
        self.observation_name_dict = self.__create_map(self.observation_space_names)

        # inital state and action
        self.reset()

    def __create_action_space(self, broken_components: List[Tuple]) -> Tuple[gym.spaces.Discrete, np.array]:
        action_space = spaces.Discrete(len(broken_components))
        action_space_names = np.empty(len(broken_components), dtype=object)
        action_space_names[:] = broken_components
        return action_space, action_space_names

    def __create_observation_space(self, broken_components: List[Tuple]) -> Tuple[gym.spaces.Discrete, np.array]:
        broken_components_names = np.empty(len(broken_components), dtype=object)
        broken_components_names[:] = broken_components
        masks = [np.array(l) for l in itertools.product([True, False], repeat=len(broken_components_names))]
        observation_names = [broken_components_names[mask] for mask in masks]
        observation_space = spaces.Discrete(len(observation_names))
        return observation_space, observation_names

    def __create_map(self, observation_space_names: List) -> Dict[str, int]:
        observation_name_dict = {}
        for i, observation_name in enumerate(observation_space_names):
            observation_name_dict[str(list(observation_name))] = i

        return observation_name_dict

    def reset(self, reward_modus: str = 'raw') -> int:
        self.current_state = 0
        self.current_state_name = list(self.observation_space_names[self.current_state])
        self.last_action = None
        self.last_action_name = None
        self.successful_action = None
        self.steps = 0
        self.reward_modus = reward_modus
        return self.current_state
    
    def set_reward_decrease_factor(factor: float) -> None:
        self.reward_decrease_factor = factor

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.steps += 1
        reward = 0
        action_name = self.action_space_names[action]
        self.successful_action = False

        if action_name in self.current_state_name:
            self.successful_action = np.random.uniform() > self.data_handler.get_repair_failure_probability(action_name)
            if self.successful_action:
                self.current_state_name.remove(action_name)
                self.current_state = self.observation_name_dict[str(self.current_state_name)]
                reward = self.__get_reward(action_name)
        else:
            if self.reward_decrease:
                reward = np.power(self.reward_decrease_factor, self.steps) * self.punishment[self.reward_modus]
            else:
                reward = self.punishment[self.reward_modus]

        if len(self.current_state_name) == 0:
            done = True
        else:
            done = False

        self.last_action = action
        self.last_action_name = action_name

        return self.current_state, reward, done, {}

    def __get_reward(self, action_name: Tuple) -> float:
        if self.reward_decrease:
            return np.power(self.reward_decrease_factor, self.steps)*self.data_handler.get_reward(action_name, type=self.reward_modus)
        else:
            return self.data_handler.get_reward(action_name, type=self.reward_modus)

    def render(self) -> None:
        print('Steps: ', self.steps)
        print('Action: ', self.last_action_name)
        print('Successful: ', self.successful_action)
        print('State: ', self.current_state_name, '\n\n')

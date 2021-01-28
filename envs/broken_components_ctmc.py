import gym
from typing import Tuple, List, Dict
from envs.broken_components import BrokenComponentsEnv
from envs.data_handler_ctmc import DataHandlerCTMC

class BrokenComponentsCTMCEnv(BrokenComponentsEnv):
    def __init__(self, broken_components: List[Tuple], reward_modus: str = 'raw', reward_decrease: bool = False,
                 reward_decrease_factor: float = 0.99, state_as_vec=False,
                 dh_data_generation: str = 'Linear', dh_take_component_id: bool = True, dh_distinguishable: bool = False,
                 transition_rate_matrix_path: str='data/transition_matrix/transition_rate_matrix_approx.csv', hidden_states: bool = True):
        super(BrokenComponentsCTMCEnv, self).__init__(broken_components, reward_modus, reward_decrease, reward_decrease_factor,
                                                      state_as_vec, dh_data_generation, dh_take_component_id, dh_distinguishable,
                                                      transition_rate_matrix_path, hidden_states)

    def create_data_handler(self):
        return DataHandlerCTMC(data_generation=self.dh_data_generation,
                               take_component_id=self.dh_take_component_id,
                               transformation=self.reward_modus,
                               distinguishable=self.dh_distinguishable,
                               transition_matrix_path=self.transition_matrix_path)

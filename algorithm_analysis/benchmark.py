import numpy as np
from envs.broken_components import DataHandler

dh = DataHandler()
dh_non_stationary = DataHandler(data_generation='ARol')
component_nums = np.arange(3, 30, 2)
seed = 42

BROKEN_COMPONENTS = [dh.get_sample_component_failure_pairs(component_num, seed) for component_num in component_nums]
BROKEN_COMPONENTS_NON_STATIONARY = [dh_non_stationary.get_sample_component_failure_pairs(component_num, seed) for component_num in component_nums]

import numpy as np
from envs.broken_components import DataHandler

dh = DataHandler()
component_nums = np.arange(3, 30, 2)
seed = 42

BROKEN_COMPONENTS_LIST = [dh.get_sample_component_failure_pairs(component_num, seed) for component_num in component_nums]
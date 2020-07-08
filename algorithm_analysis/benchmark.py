from envs.broken_components import DataHandler

dh = DataHandler()
component_nums = [4, 7, 11, 15, 29, 45, 66, 93, 124, 155, 182, 204, 220, 230, 238, 242, 245, 247, 248, 250]
seed = 42

BROKEN_COMPONENTS_LIST = [dh.get_sample_component_failure_pairs(component_num, seed) for component_num in component_nums]
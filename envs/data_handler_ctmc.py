from envs.data_handler import DataHandler
from data.transition_matrix.transition_ctmc import TransitionCTMC

class DataHandlerCTMC(DataHandler):
    
    def create_transition(self, path: str):
        return TransitionCTMC(path)
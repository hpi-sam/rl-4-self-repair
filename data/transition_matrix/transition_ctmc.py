from data.transition_matrix.transition import Transition
import pandas as pd
import copy

class TransitionCTMC(Transition):    
    def __init__(self, path_to_transition_matrix: str = 'data/transition_matrix/transition_matrix.csv'):
        self.transition_rate_matrix = self.__load_transition_matrix__(path_to_transition_matrix)
        self.transition_matrix = self.get_embedded_DTMC()
        self.mapping = self.__create_mapping_list__()

    def get_embedded_DTMC(self):
        embedded_DTMC = copy.deepcopy(self.transition_rate_matrix)
        axes = embedded_DTMC.axes
        rows = axes[0]
        columns = axes[1]
        for row in rows:
            local_row = self.transition_rate_matrix.iloc[row, 1:(len(columns) - 1)]
            local_sum = sum(local_row.values.tolist())
            for column in range(1, len(columns)):
                if local_sum > 0:
                    embedded_DTMC.iat[row, column] = self.transition_rate_matrix.iat[row, column] / local_sum
                elif row + 1 == column:
                    embedded_DTMC.iat[row, column] = 1
                else:
                    embedded_DTMC.iat[row, column] = 0
        return embedded_DTMC

    def write_embedded_DTMC(self, path: str = 'data/transition_matrix/embedded_dtmc_transition_matrix.csv'):
        self.transition_matrix.to_csv(path)
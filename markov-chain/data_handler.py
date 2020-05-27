import os
import random
import pandas as pd
from typing import Tuple, List


class DataHandler:

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()
        self.component_failure_pairs: List = []
        self.__load_data()
        self.__create_component_failure_pairs()

    def __load_data(self) -> None:
        frames = []

        # searching for all csv files in the data directory and loading the data in multiple dataframes
        for root, dirs, files in os.walk('../data'):
            for f in files:
                if f.endswith(".csv"):
                    file_path = os.path.join(root, f)
                    dataframe = pd.read_csv(file_path)
                    dataframe.columns = dataframe.columns.str.replace('\t', '')
                    frames.append(dataframe)

        # combining all dataframes to one dataframe
        self.data = pd.concat(frames, sort=False)[
            ['Optimal_Affected_Component', 'Optimal_Failure', 'Optimal_Rule', 'Optimal_Utility_Increase']]

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby(['Optimal_Affected_Component', 'Optimal_Failure']).size().reset_index().rename(
            columns={0: 'count'})
        del combinations['count']
        self.component_failure_pairs = [tuple(val) for val in combinations.values]

    def get_all_component_failure_pairs(self) -> List[Tuple]:
        return self.component_failure_pairs

    def get_sample_component_failure_pairs(self, sample_size: int) -> Tuple[Tuple]:
        if sample_size > len(self.component_failure_pairs):
            print('Error: Sample size exceeds number of (component, failure) pairs.')
        else:
            return tuple(random.sample(self.component_failure_pairs, sample_size))

    def get_reward(self, component_failure_pair: Tuple) -> float:
        component = component_failure_pair[0]
        failure = component_failure_pair[1]
        filtered = self.data[
            (self.data['Optimal_Affected_Component'] == component) & (self.data['Optimal_Failure'] == failure)]
        return filtered.sample()['Optimal_Utility_Increase'].values[0]


if __name__ == '__main__':
    dataHandler = DataHandler()
    component_failure_pairs = dataHandler.get_all_component_failure_pairs()
    print(component_failure_pairs)
    print(dataHandler.get_sample_component_failure_pairs(5))
    print(dataHandler.get_reward(component_failure_pairs[0]))

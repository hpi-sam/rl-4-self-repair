import os
import random
import pandas as pd
import numpy as np
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
        for root, dirs, files in os.walk('data'):
            for f in files:
                if f.endswith(".csv"):
                    file_path = os.path.join(root, f)
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.replace('\t', '')
                    frames.append(df)

        # combining all df to one df
        self.data = pd.concat(frames, sort=False)[
            ['Optimal_Affected_Component', 'Optimal_Failure', 'Optimal_Rule', 'Optimal_Utility_Increase']]

        # transform data
        for index, row in self.data.iterrows():
            np.seterr(divide='ignore')  # to turn off divided by zero warning
            untransformed = row['Optimal_Utility_Increase']
            self.data.loc[index, 'cube'] = np.power(untransformed, (1 / 3))
            self.data.loc[index, 'sqt'] = np.sqrt(untransformed)
            self.data.loc[index, 'log10'] = np.where(untransformed > 0, np.log10(untransformed), 0)

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby(['Optimal_Affected_Component', 'Optimal_Failure']).size().reset_index().rename(
            columns={0: 'count'})
        del combinations['count']
        self.component_failure_pairs = [tuple(val) for val in combinations.values]

    def get_all_component_failure_pairs(self) -> List[Tuple]:
        return self.component_failure_pairs

    def get_sample_component_failure_pairs(self, sample_size: int) -> List[Tuple]:
        if sample_size > len(self.component_failure_pairs):
            print('Error: Sample size exceeds number of (component, failure) pairs.')
        else:
            return list(random.sample(self.component_failure_pairs, sample_size))

    def _get_sample_row(self, component_failure_pair: Tuple) -> pd.DataFrame:
        component = component_failure_pair[0]
        failure = component_failure_pair[1]
        filtered = self.data[
            (self.data['Optimal_Affected_Component'] == component) & (self.data['Optimal_Failure'] == failure)]
        return filtered.sample()

    def get_reward(self, component_failure_pair: Tuple) -> float:
        return self._get_sample_row(component_failure_pair)['Optimal_Utility_Increase'].values[0]

    def get_reward_sqt(self, component_failure_pair: Tuple) -> float:
        return self._get_sample_row(component_failure_pair)['sqt'].values[0]

    def get_reward_cube(self, component_failure_pair: Tuple) -> float:
        return self._get_sample_row(component_failure_pair)['cube'].values[0]

    def get_reward_log10(self, component_failure_pair: Tuple) -> float:
        return self._get_sample_row(component_failure_pair)['log10'].values[0]

    def get_repair_failure_probability(self, component_failure_pair: Tuple) -> float:
        return 0.1  # static failure rate


if __name__ == '__main__':
    dataHandler = DataHandler()
    component_failure_pairs = dataHandler.get_all_component_failure_pairs()
    print(component_failure_pairs)
    print(dataHandler.get_sample_component_failure_pairs(5))
    print(dataHandler.get_reward(component_failure_pairs[0]))
    print(dataHandler.get_reward_sqt(component_failure_pairs[0]))
    print(dataHandler.get_reward_cube(component_failure_pairs[0]))
    print(dataHandler.get_reward_log10(component_failure_pairs[0]))

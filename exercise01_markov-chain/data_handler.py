import os
import pandas as pd


class DataHandler:

    data = None

    def __init__(self):
        self._load_data()

    def _load_data(self):
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
        self.data = pd.concat(frames, sort=False)[['Optimal_Affected_Component', 'Optimal_Failure', 'Optimal_Rule', 'Optimal_Utility_Increase']]

    def get_all_component_failure_pairs(self):
        combinations = self.data.groupby(['Optimal_Affected_Component', 'Optimal_Failure']).size().reset_index().rename(columns={0:'count'})
        del combinations['count']
        return combinations.values.tolist()

    def get_reward(self, component, failure):
        filtered = self.data[(self.data['Optimal_Affected_Component'] == component) & (self.data['Optimal_Failure'] == failure)]
        return filtered.sample()['Optimal_Utility_Increase'].values[0]


if __name__ == '__main__':
    dataHandler = DataHandler()
    print(dataHandler.get_all_component_failure_pairs())
    print(dataHandler.get_reward('Authentication Service', 'CF1'))

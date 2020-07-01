import os
import sys
import random
import pandas as pd
from envs.environments import check_environment
import envs.data_utils as du


class DataHandler:

    def __init__(self, data_generation: str = 'Linear', take_component_id: bool = True, type: str = 'raw'):
        '''
        Initializes the Data Handler by loading data into the environment and select between using the componenent id or name.

        :param data_generation:
            Choose between the version of the mRubis environment 'Linear' (default) | 'Saturating' | 'Combined' | 'Discontinuous'
            or shifted data 'LinearShifted'
            or non_stationary_data choose model like 'ARCH'
        :param take_component_id:
            Choose compononent id or name. When take_component_id is false, you will take name.
        :param type:
            Choose between 'raw' (Default), 'sqt', 'cube', 'log10', 'ln', 'log2'

        For all possible combinations of environment, id and type please have a look on the file 'environments.py'.
        '''

        self.environment, self.filename = check_environment(data_generation, take_component_id, type)
        self.data: pd.DataFrame = pd.DataFrame()
        self.component_failure_pairs = []
        self.__load_data()
        self.__create_component_failure_pairs()

    def __load_transform_data(self) -> None:
        frames = []

        # searching for all csv files in the data directory and loading the data in multiple dataframes
        for root, dirs, files in os.walk('data'):
            for f in files:
                if f.endswith(self.environment[0] + '.csv'):
                    file_path = os.path.join(root, f)
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.replace('\t', '')
                    frames.append(df)

        # combining all df to one df
        self.data = pd.concat(frames, sort=False)[
            ['Optimal_Affected_Component', 'Optimal_Affected_Component_Uid', 'Optimal_Failure', 'Optimal_Utility_Increase']].rename(columns={'Optimal_Utility_Increase': 'raw'})

        # transform data
        self.data = du.transform_data(self.data)

        # save transformed data this to environment as a csv file for quick reload in future
        self.data.to_csv(self.filename)

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.filename, index_col=0)[[self.environment[1], 'Optimal_Failure', self.environment[2]]]
        except FileNotFoundError:
            print('Please restart.')
            sys.exit(0)

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby([self.data.columns[0], self.data.columns[1]]).size().reset_index().rename(
            columns={0: 'count'})
        del combinations['count']
        self.component_failure_pairs = [tuple(val) for val in combinations.values]

    def get_all_component_failure_pairs(self):
        return self.component_failure_pairs

    def get_repair_failure_probability(self, component_failure_pair) -> float:
        return 0.1  # static failure rate

    def get_sample_component_failure_pairs(self, sample_size: int):
        if sample_size > len(self.component_failure_pairs):
            print('Error: Sample size exceeds number of (component, failure) pairs.')
        else:
            return list(random.sample(self.component_failure_pairs, sample_size))

    def get_reward(self, component_failure_pair) -> float:
        component = component_failure_pair[0]
        failure = component_failure_pair[1]
        filtered = self.data[
            (self.data[self.data.columns[0]] == component) & (self.data[self.data.columns[1]] == failure)]
        sample_value = 0
        if self.environment[0] in ['ARCH', 'GARCH']:
            sample_value = filtered.iloc[0][self.data.columns[2]]
            index = filtered.index[0]
            self.data = self.data.drop(index)
        else:
            sample_value = filtered.sample()[self.data.columns[2]].values[0]
        return sample_value

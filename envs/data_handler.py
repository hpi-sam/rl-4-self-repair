import os
import sys
import random
import pandas as pd
import envs.data_utils as du


class DataHandler:

    def __init__(self, environment: str = 'Linear', take_component_id: bool = True):
        '''
        Initializes the Data Handler by loading data into the environment and select between using the componenent id or name.

        :param environment:
            Choose between the version of the mRubis environment and its function for computing the utility_increase
            'Linear' (default) | 'Saturating' | 'Combined' | 'Discontinuous'
            or shifted data based on the component id
            'Linear_Id_Shifted'
        :param take_component_id:
            Choose compononent id or name. When take_component_id is false, you will take name.
        '''
        self.environment = environment
        self.component_selector = 'Optimal_Affected_Component_Uid' if take_component_id else 'Optimal_Affected_Component'
        self.data: pd.DataFrame = pd.DataFrame()
        self.component_failure_pairs = []
        self.__load_data()
        self.__create_component_failure_pairs()

    def __prepared_data_file(self) -> str:
        return 'prepared_data_' + self.environment + '.csv'

    def __load_transform_data(self) -> None:
        frames = []

        # searching for all csv files in the data directory and loading the data in multiple dataframes
        for root, dirs, files in os.walk('data'):
            for f in files:
                if f.endswith(self.environment + '.csv'):
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
        self.data.to_csv(self.__prepared_data_file())

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.__prepared_data_file(), index_col=0)
        except FileNotFoundError:
            self.__load_transform_data()

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby([self.component_selector, 'Optimal_Failure']).size().reset_index().rename(
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

    def get_reward(self, component_failure_pair, type: str ='raw') -> float:
        component = component_failure_pair[0]
        failure = component_failure_pair[1]
        filtered = self.data[
            (self.data[self.component_selector] == component) & (self.data['Optimal_Failure'] == failure)]
        return filtered.sample()[type].values[0]


if __name__ == '__main__':

    dh = DataHandler()
    component_failure_pairs = dh.get_all_component_failure_pairs()
    print('Selected mRubis environment:', dh.environment)
    print('Number of <component,failure> pairs:', len(component_failure_pairs))
    print('Get two samples:', dh.get_sample_component_failure_pairs(2))
    print('untransformed sampled:', dh.get_reward(component_failure_pairs[0]))

    # Transformation
    print('Square Root Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='sqt'))
    print('Cube Root Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='cube'))
    print('Log10 Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='log10'))
    print('Ln Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='ln'))
    print('Log2 Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='log2'))

    # Shifting
    ordering, shiftedData = du.shift_data(dh.data, 'cube', times=2)
    result = du.perform_ttest(ordering, shiftedData, 'cube')
    print('Perform T-Test on shifted data:', (len(result[result['pvalue']<0.025])))

    # Non-stationary series
    max_min_std_values = dh.data[['Optimal_Affected_Component_Uid', 'Optimal_Failure', 'raw']].groupby(['Optimal_Affected_Component_Uid', 'Optimal_Failure'])['raw'].agg([max, min, 'std']).reset_index()
    print('Create AR non-stationary time series', du.create_non_stationary_data(max_min_std_values, du.ARCH, N=100))

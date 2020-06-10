import os
import random
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List

PREPARED_DATA_FILE = 'prepared_data.csv'


class DataHandler:

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()
        self.component_failure_pairs: List = []
        self.__load_data()
        self.__create_component_failure_pairs()

    def __load_transform_data(self) -> None:
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
            ['Optimal_Affected_Component', 'Optimal_Affected_Component_Uid', 'Optimal_Failure', 'Optimal_Utility_Increase']].rename(columns={'Optimal_Utility_Increase': 'raw'})

        # transform data
        for index, row in self.data.iterrows():
            untransformed = row['raw']
            self.data.loc[index, 'cube'] = np.power(untransformed, (1 / 3))
            self.data.loc[index, 'sqt'] = np.sqrt(untransformed)
            np.seterr(divide = 'ignore')
            self.data.loc[index, 'log10'] = np.where(untransformed > 0, np.log10(untransformed), 0)
            self.data.loc[index, 'ln'] = np.where(untransformed > 0, np.log(untransformed), 0)
            self.data.loc[index, 'log2'] = np.where(untransformed > 0, np.log2(untransformed), 0)

        # save transformed data this to environment as a csv file for quick reload in future
        self.data.to_csv(PREPARED_DATA_FILE)

    def __load_data(self):
        try:
            self.data = pd.read_csv(PREPARED_DATA_FILE, index_col=0)
        except FileNotFoundError:
            self.__load_transform_data()

    def shift_data(self, data_type: str = 'raq', times: int = 1):
        mean_values = self.data.groupby(['Optimal_Affected_Component_Uid', 'Optimal_Failure'])[data_type].mean().reset_index().sort_values(by=[data_type], ascending=True)
        stdev_values = self.data.groupby(['Optimal_Affected_Component_Uid', 'Optimal_Failure'])[data_type].std().reset_index()

        data_new = self.data[['Optimal_Affected_Component_Uid', 'Optimal_Failure', data_type]].copy()
        previous = None
        for _, name in mean_values.iterrows():
            if previous is not None:
                pre_std = stdev_values.loc[(stdev_values['Optimal_Affected_Component_Uid'] == previous[0]) & (stdev_values['Optimal_Failure'] == previous[1])][data_type].tolist()[0]
                cur_std = stdev_values.loc[(stdev_values['Optimal_Affected_Component_Uid'] == name[0]) & (stdev_values['Optimal_Failure'] == name[1])][data_type].tolist()[0]
                data_new.loc[(data_new['Optimal_Affected_Component_Uid'] == name[0]) & (data_new['Optimal_Failure'] == name[1]), data_type] += (cur_std + pre_std) * times
            previous = name
        return (mean_values, data_new)

    def __create_component_failure_pairs(self) -> None:
        combinations = self.data.groupby(['Optimal_Affected_Component_Uid', 'Optimal_Failure']).size().reset_index().rename(
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

    def get_reward(self, component_failure_pair: Tuple, type: str ='raw') -> float:
        component = component_failure_pair[0]
        failure = component_failure_pair[1]
        filtered = self.data[
            (self.data['Optimal_Affected_Component_Uid'] == component) & (self.data['Optimal_Failure'] == failure)]
        return filtered.sample()[type].values[0]

    def get_repair_failure_probability(self, component_failure_pair: Tuple) -> float:
        return 0.1  # static failure rate


def perform_ttest(ordering: [Tuple], shifted_data: pd.DataFrame, type: str) -> pd.DataFrame:
    ttest_results = pd.DataFrame(columns=['a', 'b', 'statistic', 'pvalue', 'significant'])
    data_new_grouped = shifted_data.groupby(['Optimal_Affected_Component_Uid', 'Optimal_Failure'])[type].apply(list).reset_index()

    previous = None
    for index, name in ordering.iterrows():
        if previous is not None:
            pre = data_new_grouped.loc[(data_new_grouped['Optimal_Affected_Component_Uid'] == previous[0]) & (data_new_grouped['Optimal_Failure'] == previous[1])][type].tolist()[0]
            cur = data_new_grouped.loc[(data_new_grouped['Optimal_Affected_Component_Uid'] == name[0]) & (data_new_grouped['Optimal_Failure'] == name[1])][type].tolist()[0]
            result = stats.ttest_ind(pre, cur)
            new_row = pd.DataFrame({'a': str(previous), 'b': str(name), 'statistic': result[0], 'pvalue': result[1], 'significant': result[1]<0.025}, index=[0])
            ttest_results = ttest_results.append(new_row, ignore_index=True)
        previous = name

    #ttest_results[ttest_results['statistic']<-0.025][["a", "b", "pvalue"]].to_csv('ttest_results_statisticalSignificant.csv')
    #ttest_results.to_csv('ttest_' '_all.csv')
    return ttest_results


if __name__ == '__main__':
    dh = DataHandler()
    component_failure_pairs = dh.get_all_component_failure_pairs()
    print('Number of <component,failure> pairs:', len(component_failure_pairs))
    print('Get two samples:', dh.get_sample_component_failure_pairs(2))
    print('untransformed sampled:', dh.get_reward(component_failure_pairs[0]))
    print('Square Root Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='sqt'))
    print('Cube Root Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='cube'))
    print('Log10 Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='log10'))
    print('Ln Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='ln'))
    print('Log2 Transformation sampled:', dh.get_reward(component_failure_pairs[0], type='log2'))
    ordering, shiftedData = dh.shift_data('cube', times=2)
    result = perform_ttest(ordering, shiftedData, 'cube')
    print('Perform T-Test on shifted data:', (len(result[result['pvalue']<0.025])))

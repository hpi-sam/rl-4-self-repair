import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt
import pandas as pd

class HMMTrace:

    def __init__(self, iterations: int=5, sample_length: int=10000):
        self.iterations = iterations
        self.sample_length = sample_length
        self.__init_names__()
        self.__create_transition_matrix__()

    def __init_names__(self):
        components = ['Authentication Service', 'Availability Item Filter', 'Bid and Buy Service', 'Buy Now Item Filter', 'Category Item Filter', 'Comment Item Filter', 'Future Sales Item Filter', 'Inventory Service', 'Item Management Service', 'Last Second Sales Item Filter', 'Past Sales Item Filter', 'Persistence Service', 'Query Service', 'Recommendation Item Filter', 'Region Item Filter', 'Reputation Service', 'Seller Reputation Item Filter', 'User Management Service', 'Supervisory Component']
        component_states = ['operational', 'degraded', 'unresponsive']
        self.names = [''] * (19 * 3)
        for c in range(len(components)):
            for s in range(len(component_states)):
                self.names[3 * c + s] = components[c] + ' ' + component_states[s]

    def __create_transition_matrix__(self):
        A_internal = [[0.15, 0.05, 0.0, 0.8, 0.0, 0.0], [0.5, 0.25, 0.1, 0.0, 0.15, 0.0], [0.25, 0.0, 0.5, 0.125, 0.0, 0.125]]
        # the first 3 columns refer to internal transitions, the last 3 columns refer to external transition to other components

        A_external = [[0.0 for col in range(19)] for row in range(19)]
        A_external[0][12] = 1

        A_external[1][14] = 1

        A_external[2][0] = .25
        A_external[2][7] = .25
        A_external[2][11] = .25
        A_external[2][12] = .25

        A_external[3][1] = 1

        A_external[4][13] = 1

        A_external[5][4] = 1

        A_external[6][18] = 1

        A_external[7][11] = .5
        A_external[7][12] = .5

        A_external[8][0] = .33
        A_external[8][11] = .34
        A_external[8][12] = .33

        A_external[9][10] = 1

        A_external[10][3] = 1

        A_external[11][18] = 1

        A_external[12][9] = 1

        A_external[13][6] = 1

        A_external[14][16] = 1

        A_external[15][0] = .33
        A_external[15][11] = .34
        A_external[15][12] = .33

        A_external[16][5] = 1

        A_external[17][0] = .33
        A_external[17][11] = .34
        A_external[17][12] = .33

        A_external[18][0] = .17
        A_external[18][2] = .17
        A_external[18][7] = .17
        A_external[18][8] = .17
        A_external[18][15] = .16
        A_external[18][17] = .16

        A_complete = [[0.0 for col in range(19 * 3)] for row in range(19 * 3)]
        for x in range(19):
            for s in range(3):
                xs = x * 3 + s
                # internal transitions
                A_complete[xs][x * 3] = A_internal[s][0]
                A_complete[xs][x * 3 + 1] = A_internal[s][1]
                A_complete[xs][x * 3 + 2] = A_internal[s][2]
                for y in range(19):
                    if y != x:
                        # external transitions
                        A_complete[xs][y * 3] = A_internal[s][3] * A_external[x][y]
                        A_complete[xs][y * 3 + 1] = A_internal[s][4] * A_external[x][y]
                        A_complete[xs][y * 3 + 2] = A_internal[s][5] * A_external[x][y]
        T_complete = np.array(A_complete)

        O_internal = np.array([[0.85, 0.15, 0.0, 0.0, 0.0, 0.0], [0.1, 0.6, 0.2, 0.0, 0.1, 0.0], [0.05, 0.0, 0.7, 0.1, 0.0, 0.15]])
        O_complete = [[0.0 for col in range(19 * 3)] for row in range(19 * 3)]
        for x in range(19):
            for s in range(3):
                xs = x * 3 + s
                # internal transitions
                O_complete[xs][x * 3] = O_internal[s][0]
                O_complete[xs][x * 3 + 1] = O_internal[s][1]
                O_complete[xs][x * 3 + 2] = O_internal[s][2]
                for y in range(19):
                    if y != x:
                        # external transitions
                        O_complete[xs][y * 3] = O_internal[s][3] * A_external[x][y]
                        O_complete[xs][y * 3 + 1] = O_internal[s][4] * A_external[x][y]
                        O_complete[xs][y * 3 + 2] = O_internal[s][5] * A_external[x][y]
        O_complete = np.array(O_complete)

        self.model = hmm.GaussianHMM(n_components=19 * 3, covariance_type="spherical")

        startprob = [0.0] * (19 * 3)
        startprob[0] = 1.0
        self.model.startprob_ = np.array(startprob)
        self.model.transmat_ = T_complete
        self.model.features = 19 * 3

        self.model.means_ = O_complete
        covar = 0.1 ** 323
        self.model.covars_ = np.tile([covar], 3 * 19)

        self.trace, self.probs = self.model.sample(self.sample_length)
        self.trace = np.round(self.trace, 8)
        self.discrete_trace = self.create_discrete_trace()

        self.approx_transmat = self.create_transition_matrix()

    def create_discrete_trace(self):
        new_trace = []
        for i in range(len(self.trace)):
            new_trace.append([])
            probs = self.trace[i]
            chosen = np.random.choice(len(probs), 1, p=probs)[0]
            for j in range(len(probs)):
                new_trace[i].append(1 if j == chosen else 0)
        return new_trace

    def get_transformation(self, means):
        transformation = [-1] * len(means)
        for i in range(len(means)):
            max_value = 0
            max_index = -1
            mean = means[i]
            for j in range(len(mean)):
                if mean[j] > max_value:
                    max_value = mean[j]
                    max_index = j
            transformation[i] = max_index
        return transformation

    def transform(self, matrix, transformation):
        new_matrix = [[-1 for col in range(len(matrix))] for row in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_matrix[transformation[i]][transformation[j]] = matrix[i][j]
        return new_matrix

    def create_transition_matrix(self):
        remodel = hmm.GaussianHMM(n_components = 19 * 3, covariance_type='full', n_iter = self.iterations)
        remodel.fit(self.discrete_trace)
        transform_remodel = self.get_transformation(remodel.means_)
        return self.transform(remodel.transmat_, transform_remodel)

    def write_transition_matrix(self, path: str='transition_approx.csv'):
        df = pd.DataFrame(self.approx_transmat, index=self.names, columns=self.names)
        df.to_csv(path, index_label="Sources")
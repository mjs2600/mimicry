import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mutual_info_score

class Mimic(object):
    def __init__(self, bitstring_length, fitness_function):
        self.distribution = InitialDistribution(bitstring_length)
        self.sample_set = SampleSet(self.distribution, fitness_function)
        self.fitness_function = fitness_function
        self.distribution = Distribution()

    def fit(self, percentile):
        pass


class SampleSet(object):
    def __init__(self, samples, fitness_function, maximize=True):
        self.samples = samples
        self.fitness_function = fitness_function
        self.maximize = maximize

        self.complete_graph = self._generate_mutual_information_graph()

    def calculate_fitness(self):
        sorted_samples = sorted(
            self.samples,
            key=self.fitness_function,
            reverse=self.maximize,
        )
        return np.matrix(sorted_samples)

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index]

    # Consider using a property decorator here.
    def _generate_mutual_information_graph(self):
        samples = np.asarray(self.samples)
        complete_graph = nx.complete_graph(samples.shape[0])

        for edge in complete_graph.edges():
            mutual_info = mutual_info_score(samples[edge[0]], samples[edge[1]])

            complete_graph.edge[edge[0]][edge[1]]['weight'] = mutual_info

        return complete_graph






class Distribution(object):
    def __init__(self):
        pass

    def generate_samples(self):
        pass

    def generate_mutual_information_graph(self, samples):
        pass


class InitialDistribution(Distribution):
    def __init__(self, bitstring_length):
        self.bitstring_length = bitstring_length

import numpy as np


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

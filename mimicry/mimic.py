class Mimic(object):
    def __init__(self, bitstring_length, cost_function):
        self.sample_set = SampleSet(bitstring_length)
        self.cost_function = cost_function
        self.distribution = Distribution()

    def fit(self, percentile):
        pass


class SampleSet(object):
    def __init__(self, distribution):
        self.samples = distribution.generate_samples

    def generate_mutual_information_graph(self):
        pass

    def calculate_fitness(self, fitness_function):
        pass

    def get_percentile(self, percentile):
        pass


class Distribution(object):
    def __init__(self):
        pass

    def generate_samples(self):
        pass

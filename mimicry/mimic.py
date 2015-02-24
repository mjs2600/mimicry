class Mimic(object):
    def __init__(self, bitstring_length, cost_function):
        self.sample_set = SampleSet(bitstring_length)
        self.cost_function = cost_function
        self.distribution = Distribution()

    def fit(self, percentile):
        pass


class SampleSet(object):
    def __init__(self, samples, fitness_function, maximize=True):
        self.samples = samples
        self.fit_function = fitness_function
        self.maximize = maximize

    def calculate_fitness(self):
        s = [(self.fit_function(sample), sample) for sample in self.samples]
        return sorted(s, reverse=self.maximize)

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index]

    def generate_mutual_information_graph(self):
        pass


class Distribution(object):
    def __init__(self):
        pass

    def generate_samples(self):
        pass

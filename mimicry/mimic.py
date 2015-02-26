import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import mutual_info_score
import copy


class Mimic(object):
    def __init__(self, domain, fitness_function, samples=50):
        self.domain = domain
        self.samples = samples
        initial_samples = np.array(self._generate_initial_samples())
        self.sample_set = SampleSet(initial_samples, fitness_function)
        self.fitness_function = fitness_function

    def fit(self, percentile):
        samples = self.sample_set.get_percentile(percentile)
        self.distribution = Distribution(samples)

    def _generate_initial_samples(self):
        return [self._generate_initial_sample() for i in xrange(self.samples)]

    def _generate_initial_sample(self):
        return [random.randint(self.domain[i][0], self.domain[i][1])
                for i in range(len(self.domain))]


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
        return np.array(sorted_samples)

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index]


class Distribution(object):
    def __init__(self, samples):
        self.samples = samples
        self.complete_graph = self._generate_mutual_information_graph()
        self.spanning_graph = self._generate_spanning_graph()

    def generate_samples(self, domain):
        pass

    def _generate_bayes_net(self):
        # Pseudo Code
        # 1. Start at any node(probably 0 since that will be the easiest for
        # indexing)
        # 2. At each node figure out the conditional probability
        # 3. Add it to the new graph (which should probably be directed)
        # 4. Find unprocessed adjacent nodes
        # 5. If any go to 2
        #    Else return the bayes net'

        samples = np.asarray(self.samples)

        self.bayes_net = nx.bfs_tree(self.spanning_graph, 0)

        for node_ind in self.bayes_net.nodes():
            node_array = samples[:, node_ind]

            unconditional_distr = np.histogram(
                node_array,
                (np.max(node_array)+1),
            )[0] / float(node_array.shape[0])

            print(self.bayes_net.successors(node_ind))
            self.bayes_net.node[node_ind] = unconditional_distr
            print(self.bayes_net.node[node_ind])

    def _generate_spanning_graph(self):
        return nx.prim_mst(self.complete_graph)

    def _generate_mutual_information_graph(self):
        samples = np.asarray(self.samples)
        complete_graph = nx.complete_graph(samples.shape[1])

        for edge in complete_graph.edges():
            mutual_info = mutual_info_score(samples[edge[0]], samples[edge[1]])

            complete_graph.edge[edge[0]][edge[1]]['weight'] = -mutual_info

        return complete_graph


if __name__ == "__main__":
    samples = [
        [0, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
    ]

    distribution = Distribution(samples)

    distribution._generate_bayes_net()

    # self.assertEqual(
    #     expected_results,
    #     distribution.spanning_graph.edges(data=True),
    # )

    pos = nx.spring_layout(distribution.spanning_graph)

    edge_labels = dict(
        [((u, v,), d['weight'])
         for u, v, d in distribution.spanning_graph.edges(data=True)]
    )

    nx.draw_networkx(distribution.spanning_graph, pos)
    nx.draw_networkx_edge_labels(
        distribution.spanning_graph,
        pos,
        edge_labels=edge_labels,
    )

    plt.show()

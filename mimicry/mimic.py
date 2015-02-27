import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from sklearn.metrics import mutual_info_score

np.set_printoptions(precision=4)

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

    def generate_samples(self, number_to_generate):
        sample_len = len(self.bayes_net.node)
        samples = np.zeros((number_to_generate, sample_len))
        values = self.bayes_net.node[0]["probabilities"].keys()
        probabilities = self.bayes_net.node[0]["probabilities"].values()
        dist = stats.rv_discrete(name="dist", values=(values, probabilities))
        samples[:, 0] = dist.rvs(size=number_to_generate)
        for parent, current in self.bayes_net.edges_iter():
            for i in xrange(number_to_generate):
                parent_val = samples[i, parent]
                current_node = self.bayes_net.node[current]
                cond_dist = current_node["probabilities"][parent_val]
                values = cond_dist.keys()
                probabilities = cond_dist.values()
                dist = stats.rv_discrete(
                    name="dist",
                    values=(values, probabilities)
                )
                samples[i, current] = dist.rvs()

        return samples

    def _generate_bayes_net(self):
        # Pseudo Code
        # 1. Start at any node(probably 0 since that will be the easiest for
        # indexing)
        # 2. At each node figure out the conditional probability
        # 3. Add it to the new graph (which should probably be directed)
        # 4. Find unprocessed adjacent nodes
        # 5. If any go to 2
        #    Else return the bayes net'

        # Will it be possible that zero is not the root? If so, we need to pick one
        root = 0

        samples = np.asarray(self.samples)

        self.bayes_net = nx.bfs_tree(self.spanning_graph, root)


        for parent, child in self.bayes_net.edges():
            #Check if probabilities have already been calculated for parent

            if not self.bayes_net.predecessors(parent):
                parent_array = samples[:, parent]

                parent_probs = np.histogram(parent_array,
                                            (np.max(parent_array)+1),
                                            )[0] / float(parent_array.shape[0])

                self.bayes_net.node[parent] = dict(enumerate(parent_probs))

            else:
                # if this is the case, the parent already has a set of cond. probs.
                # so we need to add together the probabilities from each condition
                # to get the total probability of each value

                parent_probs=[0]*len(self.bayes_net.node[parent].keys())
                for condition, probability in self.bayes_net.node[parent].iteritems():
                    for index,_ in enumerate(parent_probs):
                        parent_probs[index] += probability[index]

            child_array = samples[:, child]
            child_probs = np.histogram(child_array,
                                       (np.max(child_array)+1),
                                       )[0] / float(child_array.shape[0])

            for condition, probability in self.bayes_net.node[parent].iteritems():
                if type(probability).__module__ == np.__name__:
                    # parent already has cond. probs, use probs calculated in "else" above
                    self.bayes_net.node[child][condition] = parent_probs[condition]*child_probs
                else:
                    self.bayes_net.node[child][condition] = probability*child_probs

        # Needed to leave probabilities as numpy arrays to facilitate calculations
        # Couldn't think of a slicker way to convert them to final form as dicts
        for key, node in self.bayes_net.node.iteritems():
            if key == root:
                continue
            for key, item in node.iteritems():
                node[key] = dict(enumerate(item))




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
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ]

    distribution = Distribution(samples)

    distribution._generate_bayes_net()

    for node_ind in distribution.bayes_net.nodes():
            print(distribution.bayes_net.node[node_ind])

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

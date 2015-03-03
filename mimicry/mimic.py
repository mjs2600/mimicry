import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from sklearn.metrics import mutual_info_score
from collections import OrderedDict

np.set_printoptions(precision=4)


class Mimic(object):
    """
    Usage: from mimicry import Mimic

    :param domain: list of tuples containing the min and max value for each parameter to be optimized, for a bit
    string, this would be [(0, 1)]*bit_string_length

    :param fitness_function: callable that will take a single instance of your optimization parameters and return
    a scalar fitness score

    :param samples: Number of samples to generate from the distribution each iteration

    :param percentile: Percentile of the distribution to keep after each iteration, default is 0.90

    """

    def __init__(self, domain, fitness_function, samples=1000, percentile=0.90):

        self.domain = domain
        self.samples = samples
        initial_samples = np.array(self._generate_initial_samples())
        self.sample_set = SampleSet(initial_samples, fitness_function)
        self.fitness_function = fitness_function
        self.percentile = percentile

    def fit(self):
        """
        Run this to perform one iteration of the Mimic algorithm

        :return: A list containing the top percentile of data points
        """

        samples = self.sample_set.get_percentile(self.percentile)
        self.distribution = Distribution(samples)
        self.sample_set = SampleSet(
            self.distribution.generate_samples(self.samples),
            self.fitness_function,
        )
        return self.sample_set.get_percentile(self.percentile)

    def _generate_initial_samples(self):
        return [self._generate_initial_sample() for i in xrange(self.samples)]

    def _generate_initial_sample(self):
        return [random.randint(self.domain[i][0], self.domain[i][1])
                for i in xrange(len(self.domain))]


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

        index = int(len(fit_samples) * (percentile))

        return fit_samples[:index]


class Distribution(object):
    def __init__(self, samples):
        self.samples = samples
        self.complete_graph = self._generate_mutual_information_graph()
        self.spanning_graph = self._generate_spanning_graph()
        self._generate_bayes_net()

    def generate_samples(self, number_to_generate):
        sample_len = len(self.bayes_net.node)
        samples = np.zeros((number_to_generate, sample_len))
        values = self.bayes_net.node[0]["probabilities"].keys()
        probabilities = self.bayes_net.node[0]["probabilities"].values()
        dist = stats.rv_discrete(name="dist", values=(values, probabilities))
        samples[:, 0] = dist.rvs(size=number_to_generate)

        try:
            print(self.bayes_net.edges())
            for parent, current in self.bayes_net.edges_iter():
                print(parent, current)
                for i in xrange(number_to_generate):
                    parent_val = samples[i, parent]
                    current_node = self.bayes_net.node[current]
                    print(self.bayes_net.node[parent])
                    # print(parent_val)
                    print(current_node)
                    print("\n")
                    cond_dist = current_node["probabilities"][int(parent_val)]
                    values = cond_dist.keys()
                    probabilities = cond_dist.values()
                    dist = stats.rv_discrete(
                        name="dist",
                        values=(values, probabilities)
                    )
                    samples[i, current] = dist.rvs()
        except KeyError:
            print("Error with parent_val %d"%parent_val)
            print(samples[:, parent])
            self.plot_bayes_net()


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

        # Will it be possible that zero is not the root? If so, we need to pick
        # one
        root = 0

        samples = np.asarray(self.samples)

        self.bayes_net = nx.bfs_tree(self.spanning_graph, root)

        for parent, child in self.bayes_net.edges():

            parent_array = samples[:, parent].astype(int)

            # print(np.unique(parent_array).shape)

            # Check if node is root
            if not self.bayes_net.predecessors(parent):


                # parent_probs = np.histogram(parent_array,
                #                             (np.unique(parent_array).shape[0]+1),
                #                             )[0] / float(parent_array.shape[0])
                #
                # print(parent_array+np.abs(np.min(parent_array)))

                binning_array = parent_array
                if np.min(binning_array) < 0:
                    binning_array += np.abs(np.min(parent_array))

                parent_probs = np.bincount(binning_array)/float(parent_array.shape[0])

                # self.bayes_net.node[parent]["probabilities"] = dict(enumerate(parent_probs))

                # print(dict(enumerate(parent_probs)))
                # print(dict(zip(list(OrderedDict.fromkeys(parent_array)), parent_probs.tolist())))

                # print(np.unique(np.sort(parent_array)))
                self.bayes_net.node[parent]["probabilities"] = dict(zip(np.unique(np.sort(parent_array)).tolist(), parent_probs.tolist()))
                # print(self.bayes_net.node[parent]["probabilities"])

            child_array = samples[:, child].astype(int)

            unique_parents = np.unique(parent_array)
            print(unique_parents)
            for parent_val in unique_parents:
                # print(parent_val)
                parent_inds = np.argwhere(parent_array == parent_val)
                sub_child = child_array[parent_inds]
                sub_child = sub_child.flatten()

                # print(sub_child)
                # print(sub_child.shape)


                # child_probs = np.histogram(sub_child,
                #                            (np.unique(child_array).shape[0]+1),
                #                            )[0] / float(sub_child.shape[0])

                binning_array = sub_child
                if np.min(binning_array) < 0:
                    binning_array += np.abs(np.min(binning_array))

                child_probs = np.bincount(binning_array)/float(sub_child.shape[0])

                # print(child_probs)

                # # If P(0) = 1 then child_probs = [1.]
                # # must append zeros to ensure output consistency
                # while child_probs.shape[0] < unique_parents.shape[0]:
                while child_probs.shape[0] < np.unique(child_array).max() + 1:
                    child_probs = np.append(child_probs, 0.)

                # self.bayes_net.node[child][parent_val] = dict(enumerate(child_probs))

                # self.bayes_net.node[child][parent_val] = dict(zip(list(dict(enumerate(child_probs))), child_probs.tolist()))
                # print(sub_child)
                # print(np.unique(np.sort(child_array)))
                # print(child_probs)
                # self.bayes_net.node[child][parent_val] = dict(zip(list(OrderedDict.fromkeys(sub_child)), child_probs.tolist()))

                self.bayes_net.node[child][parent_val] = dict(zip(np.unique(np.sort(child_array)).tolist(), child_probs[np.unique(np.sort(child_array))].tolist()))

                # print(dict(zip(np.unique(np.sort(child_array)).tolist(), child_probs[np.unique(np.sort(child_array))].tolist())))
                # print(dict(enumerate(child_probs)))


            self.bayes_net.node[child] = dict(probabilities=self.bayes_net.node[child])

            # print(self.bayes_net.node[parent])
            # print(self.bayes_net.node[child])


    def plot_bayes_net(self):
        pos = nx.spring_layout(self.spanning_graph)

        edge_labels = dict(
            [((u, v,), d['weight'])
             for u, v, d in self.spanning_graph.edges(data=True)]
        )

        nx.draw_networkx(self.spanning_graph, pos)
        nx.draw_networkx_edge_labels(
            mimic.distribution.spanning_graph,
            pos,
            edge_labels=edge_labels,
        )

        plt.show()


    def _generate_spanning_graph(self):
        return nx.prim_mst(self.complete_graph)

    def _generate_mutual_information_graph(self):
        samples = np.asarray(self.samples)
        complete_graph = nx.complete_graph(samples.shape[1])

        for edge in complete_graph.edges():
            mutual_info = mutual_info_score(
                samples[:, edge[0]],
                samples[:, edge[1]]
            )

            complete_graph.edge[edge[0]][edge[1]]['weight'] = -mutual_info

        return complete_graph


if __name__ == "__main__":
    # samples = [
    #     [1, 0, 0, 1],
    #     [1, 0, 1, 1],
    #     [0, 1, 1, 0],
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 1],
    #     [1, 0, 0, 0],
    # ]
    #
    # distribution = Distribution(samples)
    #
    # distribution._generate_bayes_net()
    #
    # for node_ind in distribution.bayes_net.nodes():
    #         print(distribution.bayes_net.node[node_ind])
    #
    # pos = nx.spring_layout(distribution.spanning_graph)
    #
    # edge_labels = dict(
    #     [((u, v,), d['weight'])
    #      for u, v, d in distribution.spanning_graph.edges(data=True)]
    # )
    #
    # nx.draw_networkx(distribution.spanning_graph, pos)
    # nx.draw_networkx_edge_labels(
    #     distribution.spanning_graph,
    #     pos,
    #     edge_labels=edge_labels,
    # )
    #
    # plt.show()

    mimic = Mimic([(0,10)]*8,sum, samples=25, percentile=1.)

    # mimic.fit()
    #
    # mimic.distribution._generate_bayes_net()
    #
    # # for node_ind in mimic.distribution.bayes_net.nodes():
    # #         print(distribution.bayes_net.node[node_ind])
    #
    # pos = nx.spring_layout(mimic.distribution.spanning_graph)
    #
    # edge_labels = dict(
    #     [((u, v,), d['weight'])
    #      for u, v, d in mimic.distribution.spanning_graph.edges(data=True)]
    # )
    #
    # nx.draw_networkx(mimic.distribution.spanning_graph, pos)
    # nx.draw_networkx_edge_labels(
    #     mimic.distribution.spanning_graph,
    #     pos,
    #     edge_labels=edge_labels,
    # )
    #
    # plt.show()

    results_mimic =[]
    for i in range(50):
        # print(i)
        results_mimic.append(mimic.fit()[0])



    results_mimic = [sum(results) for results in results_mimic]

    results_mimic = np.asarray(results_mimic).flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(results_mimic, label='Mimic')


    plt.show()




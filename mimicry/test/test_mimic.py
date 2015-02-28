import unittest
import random
from mimicry import mimic
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np


class TestMimic(unittest.TestCase):
    def test_inital_samples(self):
        random.seed(0)

        domain = [(0, 1)] * 10
        m = mimic.Mimic(domain, sum, samples=100)
        expected_results = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        ]
        top_samples = m.sample_set.get_percentile(.04)
        self.assertTrue(np.equal(top_samples, expected_results).all())

    def test_mimic(self):
        domain = [(0, 1)] * 7
        m = mimic.Mimic(domain, sum, samples=100)
        for i in xrange(25):
            # print np.average([sum(sample) for sample in m.fit()[:5]])
            m.fit()
        results = m.fit()
        self.assertTrue(results[0].all())


class TestSampleSet(unittest.TestCase):
    def test_get_precentile(self):
        samples = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        expected_results = np.matrix([
            [1, 1, 1],
            [0, 1, 1],
        ])
        sample_set = mimic.SampleSet(samples, sum)

        self.assertTrue(
            np.equal(sample_set.get_percentile(0.5), expected_results).all()
        )


class TestDistribution(unittest.TestCase):
    def test__generate_mutual_information_graph(self):
        samples = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

        expected_results = [
            (0, 1, {'weight': -0.21576155433883565}),
            (0, 2, {'weight': -0.084949518397698542}),
            (1, 2, {'weight': -0.21576155433883565}),
        ]

        distribution = mimic.Distribution(samples)

        self.assertEqual(
            expected_results,
            distribution.complete_graph.edges(data=True),
        )

        # pos = nx.spring_layout(sample_set.complete_graph)

        # edge_labels=dict([((u,v,),d['weight'])
        # for u,v,d in sample_set.complete_graph.edges(data=True)])

        # nx.draw_networkx(sample_set.complete_graph, pos)
        # nx.draw_networkx_edge_labels(sample_set.complete_graph,pos,
        # edge_labels=edge_labels)

        # plt.show()

    def test__generate_spanning_graph(self):
        samples = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

        expected_results = [
            (0, 1),
            (1, 2),
        ]

        distribution = mimic.Distribution(samples)

        self.assertEqual(
            expected_results,
            distribution.spanning_graph.edges(),
        )

        # pos = nx.spring_layout(sample_set.complete_graph)

        # edge_labels=dict([((u,v,),d['weight'])
        # for u,v,d in sample_set.complete_graph.edges(data=True)])

        # nx.draw_networkx(sample_set.complete_graph, pos)
        # nx.draw_networkx_edge_labels(sample_set.complete_graph,pos,
        # edge_labels=edge_labels)

        # plt.show()

    def test_generate_samples(self):
        expected_results = np.array([[1, 0, 1, 1]])
        graph = nx.DiGraph()

        graph.add_node(0, probabilities={0: 0, 1: 1})
        graph.add_node(1, probabilities={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}})
        graph.add_node(2, probabilities={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}})
        graph.add_node(3, probabilities={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}})

        graph.add_edges_from([
            (0, 1),
            (1, 2),
            (1, 3),
        ])

        distribution = mimic.Distribution([[0, 0], [0, 0]])
        distribution.bayes_net = graph

        self.assertTrue(np.equal(
            expected_results,
            distribution.generate_samples(1),
        ).all())


if __name__ == '__main__':
        unittest.main()

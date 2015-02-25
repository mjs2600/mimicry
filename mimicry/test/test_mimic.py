import unittest
from mimicry import mimic
import networkx as nx
import matplotlib.pyplot as plt


class TestSampleSet(unittest.TestCase):
    def test_get_precentile(self):
        samples = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        expected_results = [
            (3, [1, 1, 1]),
            (2, [0, 1, 1]),
        ]
        sample_set = mimic.SampleSet(samples, sum)

        self.assertEqual(sample_set.get_percentile(0.5), expected_results)

    def test__generate_mutual_information_graph(self):
        samples = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

        sample_set = mimic.SampleSet(samples, sum)

        pos = nx.spring_layout(sample_set.complete_graph)

        edge_labels=dict([((u,v,),d['weight']) for u,v,d in sample_set.complete_graph.edges(data=True)])

        nx.draw_networkx(sample_set.complete_graph, pos)
        nx.draw_networkx_edge_labels(sample_set.complete_graph,pos, edge_labels=edge_labels)

        plt.show()


if __name__ == '__main__':
        unittest.main()

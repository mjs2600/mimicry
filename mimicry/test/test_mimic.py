import unittest
from mimicry import mimic
import numpy as np


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

if __name__ == '__main__':
        unittest.main()

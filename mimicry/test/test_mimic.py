import unittest
from mimicry import mimic


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

if __name__ == '__main__':
        unittest.main()

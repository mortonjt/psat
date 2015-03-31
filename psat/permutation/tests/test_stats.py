import numpy as np
import unittest
from psat.permutation.stats import ttest


class TestStats(unittest.TestCase):
    def test_ttest(self):

        N = 10
        mat = np.vstack((
            np.hstack((np.random.random(N/2),
                       np.random.random(N/2) + 100)),
            np.random.random(N),
            np.random.random(N),
            np.random.random(N),
            np.random.random(N)))

        cats = np.array([0] * (N/2) + [1] * (N/2), dtype=np.float32)
        t_stats, pvalues = ttest(mat, cats, 1000)
        print t_stats
        print pvalues


if __name__ == "__main__":
    unittest.main()

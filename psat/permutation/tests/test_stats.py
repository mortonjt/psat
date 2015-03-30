import numpy as np
import scipy as sp
import unittest

class TestStats(unittest.TestCase):
    def test_ttest(self):

        ## Large test
        N = 10
        mat = np.array(
            np.matrix(np.vstack((
                np.hstack(np.random.random(N/2),
                          np.random.random(N/2) + 100)
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N), dtype=np.float32))

        cats = np.array([0] * (N/2) + [1] * (N/2), dtype=np.float32)
        t_stats, pvalues = ttest(mat, cats, 1000)


if __name__=="__main__":
    unittest.main()

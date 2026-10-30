import unittest

import numpy as np
from scipy import sparse

from midterm.src.models import personalized_pagerank


class TestPageRank(unittest.TestCase):
    def test_cycle_reference(self):
        W = sparse.csr_matrix(np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]))
        alpha = 0.85
        p, err, _ = personalized_pagerank(W.T.tocsr(), np.array([1., 0., 0.]), alpha,
                                          np.array([False, False, False]), iterations=200, tol=1e-14)
        expected = (1 - alpha) / (1 - alpha ** 3) * np.array([1, alpha, alpha ** 2])
        self.assertTrue(np.allclose(p, expected, atol=1e-10))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertLess(err, 1e-12)

    def test_dangling_mass_returns_to_restart(self):
        W = sparse.csr_matrix(np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
        p, err, _ = personalized_pagerank(W.T.tocsr(), np.array([1., 0., 0.]), 0.85,
                                          np.array([False, False, True]), iterations=200, tol=1e-14)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertLess(err, 1e-12)
        self.assertGreater(float(p[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
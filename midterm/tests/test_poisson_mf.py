import unittest
import numpy as np
from scipy import sparse
from midterm.src.poisson_mf import PoissonMF

class TestPoissonMF(unittest.TestCase):
    def test_objective_counts_cold_and_determinism(self):
        X=sparse.csr_matrix([[12.,1.,0.],[1.,10.,0.],[0.,0.,0.]])
        a=PoissonMF().fit(X,k=2,iterations=100)
        b=PoissonMF().fit(X,k=2,iterations=100)
        self.assertTrue(np.all(np.diff(a.losses)<=1e-7))
        prediction=a.score(np.arange(3),np.arange(3))
        self.assertGreater(prediction[0,0],prediction[0,1])
        self.assertGreater(prediction[1,1],prediction[1,0])
        self.assertTrue(np.all(prediction[2]==0));self.assertTrue(np.all(prediction[:,2]==0))
        self.assertTrue(np.array_equal(a.U,b.U))
        extended=PoissonMF().fit(sparse.csr_matrix(np.pad(X.toarray(),((0,1),(0,1)))),k=2,iterations=100)
        self.assertTrue(np.array_equal(a.U[:2],extended.U[:2]))

if __name__=='__main__':unittest.main()

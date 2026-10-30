import unittest
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from midterm.src.frequency_association import RegularizedAssociation

class TestRegularizedAssociation(unittest.TestCase):
    def test_support_penalty_and_future_population_invariance(self):
        x=np.array([[1,1,0],[1,1,0],[1,0,1],[0,0,1]],dtype=float)
        cfg={'association':{'min_support':1,'min_confidence':0.,'min_lift':0.,'top_k':10,'lift_population':'training_active'}}
        a=RegularizedAssociation().fit_regularized(SimpleNamespace(X_binary=sparse.csr_matrix(x)),cfg,10.,.5)
        self.assertAlmostEqual(a.S[0,1],(2/3)*(2/12)*np.sqrt(4/2))
        b=RegularizedAssociation().fit_regularized(SimpleNamespace(X_binary=sparse.csr_matrix(np.vstack([x,np.zeros(3)]))),cfg,10.,.5)
        self.assertTrue(np.array_equal(a.S.toarray(),b.S.toarray()))
        self.assertTrue(np.all(np.isfinite(a.S.data)))

if __name__=='__main__':unittest.main()

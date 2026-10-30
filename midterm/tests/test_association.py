import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from midterm.src.models import AssociationModel


class TestAssociationTrainingPopulation(unittest.TestCase):
    def test_future_only_indexed_user_does_not_change_lift(self):
        # Two training users; the third is present only in the shared index.
        x = sparse.csr_matrix(np.array([[1, 1], [0, 1], [0, 0]], dtype=float))
        data = SimpleNamespace(X_binary=x)
        cfg = {"association": {
            "min_support": 1, "min_confidence": 0.0,
            "min_lift": 1.25, "top_k": 10,
        }}
        model = AssociationModel().fit(data, cfg, seed=42)
        self.assertEqual(model.n_users, 2)
        self.assertEqual(model.S.nnz, 0)

        # Diagnostic legacy mode shows the exact denominator difference.
        legacy_cfg = {"association": {**cfg["association"],
                                       "lift_population": "all_indexed_legacy"}}
        legacy = AssociationModel().fit(data, legacy_cfg, seed=42)
        self.assertEqual(legacy.n_users, 3)
        self.assertGreater(legacy.S.nnz, 0)


if __name__ == "__main__":
    unittest.main()

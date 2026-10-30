import unittest

from midterm.src.metrics import ap_at_k, evaluate_recommendations


class TestMetrics(unittest.TestCase):
    def test_hand_ap(self):
        # ranks 1 and 3 are relevant: (1 + 2/3) / min(2, 5)
        self.assertAlmostEqual(ap_at_k([1, 2, 3], {1, 3}, k=5), (1 + 2 / 3) / 2)

    def test_duplicate_does_not_inflate(self):
        # the duplicate consumes rank 2 but cannot be a second hit
        self.assertAlmostEqual(ap_at_k([1, 1, 3], {1, 3}, k=5), (1 + 2 / 3) / 2)
        self.assertLess(ap_at_k([1, 1, 3], {1, 3}, k=5), ap_at_k([1, 3], {1, 3}, k=5))

    def test_recall_denominator_is_all_relevant(self):
        out = evaluate_recommendations({1: [1, 2]}, {1: set(range(1, 11))}, [1], k=5, n_candidates=100)
        self.assertAlmostEqual(out["recall5_eligible"], 0.2)

    def test_empty_gt_handling(self):
        out = evaluate_recommendations(
            {1: [1, 2, 3, 4, 5], 2: [1]}, {1: {1}, 2: set()}, [1, 2], k=5, n_candidates=10
        )
        self.assertEqual(out["n_eligible"], 1)
        self.assertAlmostEqual(out["map5_eligible"], 1.0)
        self.assertAlmostEqual(out["map5_all"], 0.5)

    def test_k_denominator(self):
        self.assertAlmostEqual(ap_at_k([7, 8, 9, 10, 11], {7}, k=5), 1.0)


if __name__ == "__main__":
    unittest.main()
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from midterm.src.evaluate import recommend_from_scores
from midterm.src.raw_context import build_opportunity, fit_context_profile, score_exposure


class TestRawContext(unittest.TestCase):
    def test_cutoff_and_mean_airing_exposure(self):
        start = pd.Timestamp("2020-01-01")
        cutoff = pd.Timestamp("2020-01-08")
        seconds = lambda x: int(pd.Timestamp(x).timestamp())
        profile = fit_context_profile(
            uidx=np.array([0, 0, 0, 1, 0], dtype=np.int32),
            channel=np.array([10, 10, 10, 20, 20], dtype=np.int32),
            start_ts=np.array([seconds("2020-01-02 20:00"), seconds("2020-01-03 20:00"),
                               seconds("2020-01-04 20:00"), seconds("2020-01-05 20:00"),
                               seconds("2020-01-09 20:00")]),
            stop_ts=np.array([seconds("2020-01-02 20:10"), seconds("2020-01-03 20:10"),
                              seconds("2020-01-04 20:10"), seconds("2020-01-05 20:10"),
                              seconds("2020-01-09 20:10")]),
            hour=np.array([20, 20, 20, 20, 20], dtype=np.uint8),
            weekday=np.array([3, 4, 5, 6, 3], dtype=np.uint8),
            user_ids=np.array([1, 2], dtype=np.uint64),
            channel_ids=np.array([10, 20], dtype=np.int32),
            fit_start=start, cutoff=cutoff, alpha=20,
        )
        self.assertEqual(profile.n_training_views, 4)
        self.assertEqual(profile.n_user_views.tolist(), [3, 1])
        self.assertEqual(profile.channel_affinity[1, 0], 0)
        sched = pd.DataFrame([
            (1, 10, "2020-01-08 20:00", 600),
            (1, 10, "2020-01-09 20:00", 600),
            (2, 20, "2020-01-08 20:00", 600),
            (0, 10, "2020-01-08 20:00", 600),
        ], columns=["tv_show_id", "channel_id", "start", "duration"])
        sched["start"] = pd.to_datetime(sched["start"])
        opp = build_opportunity(sched, np.array([1, 2]), np.array([10, 20]),
                                cutoff, pd.Timestamp("2020-01-10"))
        self.assertEqual(opp.n_airings.tolist(), [2, 1])
        self.assertTrue(np.allclose(np.asarray(opp.channel.sum(axis=1)).ravel(), [1, 1]))
        scores = score_exposure(profile, opp, np.array([0, 1]))
        self.assertGreater(scores["channel"][0, 0], scores["channel"][0, 1])
        self.assertGreater(scores["channel"][1, 1], scores["channel"][1, 0])

    def test_raw_only_user_can_use_context_with_shared_fallback(self):
        data = SimpleNamespace(user_ids=np.array([1, 2], dtype=np.uint64),
                               X_binary=sparse.csr_matrix((2, 5)))
        recs, stats = recommend_from_scores(
            scores=np.array([[0.0, 2.0, 0.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0, 0.0, 0.0]]),
            users=np.array([1, 2], dtype=np.uint64),
            candidate_ids=np.array([10, 20, 30, 40, 50]),
            data=data, repeat_policy="allow",
            fallback_scores=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            cold_mask=np.array([False, True]),
        )
        self.assertEqual(recs[1][0], 20)
        self.assertEqual(recs[2], [10, 20, 30, 40, 50])
        self.assertEqual(stats["cold_users"], 1)


if __name__ == "__main__":
    unittest.main()

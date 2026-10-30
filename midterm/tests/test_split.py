import unittest

import numpy as np
import pandas as pd

from midterm.src.evaluate import build_eval_data, build_train_data


class TestCutoffSplit(unittest.TestCase):
    def test_crossing_observations_are_excluded(self):
        rows = [
            # crosses training cutoff -> excluded from training
            (1, 10, 1, "2020-01-01 23:00", "2020-01-02 01:00", 7200, 1.0),
            # fully in training
            (1, 11, 1, "2020-01-01 10:00", "2020-01-01 11:00", 3600, 1.0),
            # fully in evaluation
            (2, 10, 1, "2020-01-02 10:00", "2020-01-02 11:00", 3600, 1.0),
            # crosses evaluation end -> excluded from evaluation
            (2, 12, 1, "2020-01-02 23:00", "2020-01-03 01:00", 7200, 1.0),
        ]
        cols = ["user_id", "item_id", "channel_id", "overlap_start", "overlap_end",
                "overlap_seconds", "overlap_fraction"]
        joined = pd.DataFrame(rows, columns=cols)
        for c in ("overlap_start", "overlap_end"):
            joined[c] = pd.to_datetime(joined[c])
        joined["view_start"] = joined["overlap_start"]
        cfg = {"seed": 42, "join": {"min_overlap_seconds": 60, "min_overlap_fraction": 0.0}}
        meta = pd.DataFrame(columns=["title", "category", "genres", "year", "director",
                                     "actors", "channels", "n_airings"])
        user_ids = np.array([1, 2], dtype=np.int64)
        item_ids = np.array([10, 11, 12], dtype=np.int64)
        data = build_train_data(cfg, joined, user_ids, item_ids, meta, pd.Timestamp("2020-01-02"))
        self.assertEqual(set(data.interactions["item_id"]), {11})
        self.assertEqual(int(data.X_binary.nnz), 1)
        schedule = pd.DataFrame([
            (10, "2020-01-02 10:00", 3600, 10),
            (12, "2020-01-02 23:00", 7200, 12),
        ], columns=["tv_show_id", "start", "duration", "channel_id"])
        schedule["start"] = pd.to_datetime(schedule["start"])
        ev = build_eval_data(cfg, joined, schedule, user_ids, item_ids, meta,
                             pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))
        self.assertEqual(set(ev["gt"].get(2, set())), {10})
        self.assertNotIn(12, ev["gt"].get(2, set()))


if __name__ == "__main__":
    unittest.main()
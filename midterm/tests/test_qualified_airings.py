import unittest

import numpy as np
import pandas as pd

from midterm.src.qualified_airings import qualify_airings, top5_frequency


class TestQualifiedAirings(unittest.TestCase):
    def test_union_and_eighty_percent_boundary(self):
        start = pd.Timestamp("2020-05-01 10:00")
        def row(a, b, item=10, user=1):
            return {"user_id": np.uint64(user), "item_id": item, "channel_id": 3,
                    "sched_start": start, "sched_end": start + pd.Timedelta(seconds=100),
                    "overlap_start": start + pd.Timedelta(seconds=a),
                    "overlap_end": start + pd.Timedelta(seconds=b),
                    "view_start": start + pd.Timedelta(seconds=a),
                    "view_end": start + pd.Timedelta(seconds=b)}
        joined = pd.DataFrame([row(0, 50), row(30, 80), row(0, 79, 11)])
        q = qualify_airings(joined, start, start + pd.Timedelta(days=1))
        self.assertEqual(q["item_id"].tolist(), [10])
        self.assertEqual(q["watched_seconds"].tolist(), [80])
        self.assertEqual(q["screen_time"].tolist(), [0.8])

    def test_frequency_and_tie(self):
        q = pd.DataFrame({"user_id": np.array([1]*6, dtype=np.uint64),
                          "item_id": [5, 5, 7, 7, 3, 9],
                          "channel_id": [1]*6,
                          "sched_start": pd.date_range("2020-01-01", periods=6)})
        gt, ranked = top5_frequency(q)
        self.assertEqual(ranked["item_id"].tolist(), [5, 7, 3, 9])
        self.assertEqual(gt[1], {3, 5, 7, 9})

    def test_short_airing_and_adjacent_large_uint64_users(self):
        start = pd.Timestamp("2020-05-01 10:00")
        big = np.uint64(2**64 - 3)
        j = pd.DataFrame({
            "user_id": np.array([big, np.uint64(2**64 - 2)], dtype=np.uint64),
            "item_id": [15, 15], "channel_id": [3, 3],
            "sched_start": [start, start],
            "sched_end": [start + pd.Timedelta(seconds=60)]*2,
            "overlap_start": [start, start + pd.Timedelta(seconds=12)],
            "overlap_end": [start + pd.Timedelta(seconds=48), start + pd.Timedelta(seconds=59)],
            "view_start": [start, start + pd.Timedelta(seconds=12)],
            "view_end": [start + pd.Timedelta(seconds=48), start + pd.Timedelta(seconds=59)],
        })
        q = qualify_airings(j, start, start + pd.Timedelta(days=1))
        self.assertEqual(q["user_id"].tolist(), [int(big)])
        self.assertEqual(q["watched_seconds"].tolist(), [48])


if __name__ == "__main__":
    unittest.main()

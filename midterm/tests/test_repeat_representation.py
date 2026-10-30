import unittest

import numpy as np
import pandas as pd

from midterm.src.repeat_representation_experiment import repeat_strengths, strength_scores


class TestRepeatRepresentation(unittest.TestCase):
    def test_cap_days_weeks_and_cutoff(self):
        starts = pd.to_datetime(["2020-01-06 10:00", "2020-01-06 11:00",
                                 "2020-01-07 10:00", "2020-01-13 10:00",
                                 "2020-01-14 10:00"])
        j = pd.DataFrame({
            "user_id": np.array([1]*5, dtype=np.uint64), "item_id": [10]*5,
            "view_start": starts,
            "view_end": starts + pd.Timedelta(minutes=10),
            "overlap_end": starts + pd.Timedelta(minutes=10),
            "overlap_seconds": [3000, 3000, 600, 600, 600],
            "overlap_fraction": [0.5]*5,
        })
        cutoff = pd.Timestamp("2020-01-14")
        agg = repeat_strengths(j, pd.Timestamp("2020-01-01"), cutoff)
        self.assertEqual(agg["capped_seconds"].tolist(), [4800])
        self.assertEqual(agg["days"].tolist(), [3])
        self.assertEqual(agg["weeks"].tolist(), [2])
        scores = strength_scores(agg, np.array([1], dtype=np.uint64),
                                 np.array([10, 20]), cutoff)
        self.assertAlmostEqual(scores["repeat_capped_minutes"][0, 0], np.log1p(80)*0.5**(14/24/14))
        self.assertEqual(scores["repeat_distinct_days"][0, 1], 0)


if __name__ == "__main__":
    unittest.main()

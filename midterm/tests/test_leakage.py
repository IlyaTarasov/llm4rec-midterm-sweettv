import unittest

import numpy as np
import pandas as pd

from midterm.src.data import build_item_metadata, candidate_items
from midterm.src.evaluate import build_candidate_data


class TestMetadataCutoff(unittest.TestCase):
    def test_future_metadata_is_not_used(self):
        sched = pd.DataFrame([
            (1, "2020-01-01 10:00", "2020-01-01 11:00", 3600, 1, "old", "news", "", "", "", "2020", "", ""),
            (1, "2020-02-01 10:00", "2020-02-01 11:00", 3600, 1, "new", "future", "", "", "", "2020", "", ""),
            (1, "2020-02-01 12:00", "2020-02-01 13:00", 3600, 2, "future2", "future", "", "", "", "2020", "", ""),
        ], columns=["channel_id", "start", "end", "duration", "tv_show_id", "tv_show_title",
                    "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
                    "year_of_production", "director", "actors"])
        for c in ("start", "end"):
            sched[c] = pd.to_datetime(sched[c])
        meta = build_item_metadata([sched], pd.Timestamp("2020-01-15"))
        self.assertEqual(list(meta.index), [1])
        self.assertEqual(meta.loc[1, "title"], "old")
        self.assertEqual(meta.loc[1, "n_airings"], 1)
        # Availability in the later window may use future schedule rows; metadata may not.
        cand = candidate_items(sched, pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01"))
        self.assertEqual(list(cand), [1, 2])
        # The final-stage candidate builder must not expose held-out labels.
        pool = build_candidate_data(
            sched, np.array([1, 2, 3], dtype=np.int64), meta,
            pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01"),
        )
        self.assertEqual(list(pool["candidate_ids"]), [1, 2])
        self.assertNotIn("gt", pool)
        self.assertNotIn("eval_interactions", pool)


if __name__ == "__main__":
    unittest.main()
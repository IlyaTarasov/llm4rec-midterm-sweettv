import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from midterm.src.config import resolve
from midterm.src.data import candidate_items
from midterm.src.fourth_pass import opportunity_factor


class TestFourthPass(unittest.TestCase):
    def test_opportunity_rate_shrink_and_cap(self):
        def frame(rows):
            d = pd.DataFrame(rows, columns=["tv_show_id", "channel_id", "start", "duration"])
            d["start"] = pd.to_datetime(d["start"])
            d["end"] = d["start"] + pd.to_timedelta(d["duration"], unit="s")
            return d
        past = frame([(1, 1, "2020-01-02", 60)] +
                     [(3, 1, f"2020-01-{day:02d}", 60) for day in range(1, 11)])
        future = frame([(1, 1, "2020-01-12", 60), (1, 1, "2020-01-13", 60)] +
                       [(2, 1, f"2020-01-{day:02d}", 60) for day in range(11, 21)])
        p = {"schedules": {"export_arh_11-20-final.csv": past,
                            "export_arh_21-30-final.csv": future}}
        w = {"fit_start": "2020-01-01", "cutoff": "2020-01-11",
             "end": "2020-01-21", "schedule": "21-30"}
        factors = opportunity_factor(p, w, np.array([1, 2, 3]))
        self.assertAlmostEqual(factors[0], np.sqrt(1.5))
        self.assertEqual(factors[1], 1.5)
        self.assertEqual(factors[2], 0.5)

    def test_candidate_hash_template_and_valid_ids(self):
        path = resolve("midterm/submission_candidate_opportunity.csv")
        manifest = json.loads(resolve("midterm/results/continuation/fourth_candidate_manifest.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["csv_sha256"])
        self.assertTrue(manifest["deterministic_regeneration_match"])
        d = pd.read_csv(path, dtype={"user_id": "uint64", "tv_show_id": "string"})
        template = pd.read_csv(resolve("midterm/data/sweettv-tv-program-recommender/submission.csv"), dtype={"user_id": "uint64"})
        self.assertTrue(np.array_equal(d["user_id"].to_numpy(dtype=np.uint64), template["user_id"].to_numpy(dtype=np.uint64)))
        s = pd.read_pickle(resolve("midterm/artifacts/schedule_export_arh_31-42-final.pkl"))
        allowed = set(map(int, candidate_items(s, pd.Timestamp("2020-07-27"), pd.Timestamp("2020-10-20"))))
        self.assertEqual(len(d), 1260)
        for cell in d["tv_show_id"]:
            ids = list(map(int, cell.split()))
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertTrue(all(item in allowed and item > 0 for item in ids))


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from midterm.src.config import resolve
from midterm.src.data import candidate_items


class TestCorrectedCandidate(unittest.TestCase):
    def test_template_candidates_and_manifest(self):
        path = resolve("midterm/submission_candidate_screen80.csv")
        manifest = json.loads(resolve("midterm/results/continuation/corrected_candidate_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["csv_sha256"])
        self.assertEqual(manifest["association_lift_population"], "training_active")
        self.assertTrue(manifest["second_generation_exact_match"])
        df = pd.read_csv(path, dtype={"user_id": "uint64", "tv_show_id": "string"})
        template = pd.read_csv(resolve("midterm/data/sweettv-tv-program-recommender/submission.csv"), dtype={"user_id": "uint64"})
        self.assertTrue(np.array_equal(df["user_id"].to_numpy(dtype=np.uint64), template["user_id"].to_numpy(dtype=np.uint64)))
        schedule = pd.read_pickle(resolve("midterm/artifacts/schedule_export_arh_31-42-final.pkl"))
        allowed = set(map(int, candidate_items(schedule, pd.Timestamp("2020-07-27"), pd.Timestamp("2020-10-20"))))
        self.assertEqual(len(df), 1260)
        for cell in df["tv_show_id"]:
            ids = list(map(int, cell.split()))
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertTrue(all(item in allowed and item > 0 for item in ids))


if __name__ == "__main__":
    unittest.main()

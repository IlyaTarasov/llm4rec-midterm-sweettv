import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from midterm.src.config import load_config
from midterm.src.data import candidate_items, read_submission, window_bounds


class TestSubmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.path = Path("midterm/submission.csv")
        if not cls.path.exists():
            raise unittest.SkipTest("final submission not generated yet")
        cls.sub = pd.read_csv(cls.path, dtype={"user_id": "uint64", "tv_show_id": "string"},
                              keep_default_na=False)
        cls.template = read_submission("midterm/data/sweettv-tv-program-recommender/submission.csv")

    def test_user_id_is_unsigned(self):
        self.assertEqual(str(self.sub["user_id"].dtype), "uint64")
        self.assertEqual(int((self.sub["user_id"] < 0).sum()), 0)
        self.assertEqual(len(self.sub), len(self.template))

    def test_order_and_tokens(self):
        self.assertTrue(np.array_equal(self.sub["user_id"].to_numpy(),
                                       self.template["user_id"].to_numpy()))
        parsed = [[int(x) for x in s.split()] for s in self.sub["tv_show_id"]]
        self.assertTrue(all(len(v) == 5 for v in parsed))
        self.assertTrue(all(len(set(v)) == 5 for v in parsed))
        self.assertTrue(all(all(x > 0 for x in v) for v in parsed))

    def test_all_ids_are_test_candidates(self):
        b = window_bounds(self.cfg)
        schedule = pd.read_pickle(Path(self.cfg["artifacts_dir"]) / "schedule_export_arh_31-42-final.pkl")
        allowed = set(int(x) for x in candidate_items(schedule, b["audit_end"], b["test_end"]))
        parsed = [[int(x) for x in s.split()] for s in self.sub["tv_show_id"]]
        self.assertTrue(all(all(x in allowed for x in v) for v in parsed))

    def test_sha_matches_manifest(self):
        h = hashlib.sha256(self.path.read_bytes()).hexdigest()
        manifest = json.loads(Path("midterm/results/final_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(h, manifest["submission_sha256"])


if __name__ == "__main__":
    unittest.main()
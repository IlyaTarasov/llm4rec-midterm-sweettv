import csv
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from midterm.src.config import load_config, resolve
from midterm.src.data import candidate_items
from midterm.src.evaluate import build_train_data, load_prepared
from midterm.src.metrics import topk_items
from midterm.src.models import PopularityModel


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TestFollowupCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_path = resolve("midterm/configs/candidate_repeat1.json")
        cls.cfg = load_config(cls.cfg_path)
        cls.path = resolve(cls.cfg["output"])
        cls.manifest = json.loads(resolve("midterm/results/followup/candidate_manifest.json").read_text(encoding="utf-8"))
        with resolve(cls.cfg["template"]).open(encoding="utf-8", newline="") as f:
            cls.template_users = [row["user_id"] for row in csv.DictReader(f)]
        with cls.path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cls.columns = reader.fieldnames
            cls.rows = list(reader)
        cls.p = load_prepared(cls.cfg)
        lo, hi = (pd.Timestamp(x) for x in cls.cfg["test_window"])
        cls.allowed = set(map(int, candidate_items(
            cls.p["schedules"]["export_arh_31-42-final.csv"], lo, hi
        )))

    def test_manifest_and_baseline_hashes(self):
        self.assertEqual(digest(self.path), self.manifest["candidate_sha256"])
        self.assertEqual(digest(self.cfg_path), self.manifest["configuration_sha256"])
        self.assertEqual(digest(resolve("midterm/submission.csv")),
                         self.manifest["baseline_sha256"]["submission.csv"])
        self.assertEqual(digest(resolve("midterm/configs/default.json")),
                         self.manifest["baseline_sha256"]["default_config.json"])

    def test_template_order_and_all_ids(self):
        self.assertEqual(self.columns, ["user_id", "tv_show_id"])
        self.assertEqual([row["user_id"] for row in self.rows], self.template_users)
        self.assertEqual(len(self.rows), 1260)
        for row in self.rows:
            ids = [int(x) for x in row["tv_show_id"].split()]
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertTrue(all(x > 0 and x in self.allowed for x in ids))

    def test_training_only_cold_fallback(self):
        cutoff = pd.Timestamp(self.cfg["train_window"][1])
        self.assertLess(self.p["joined"]["view_end"].max(), cutoff)
        data = build_train_data(self.cfg, self.p["joined"], self.p["user_ids"],
                                self.p["item_ids"], pd.DataFrame(), cutoff)
        users = np.array([int(row["user_id"]) for row in self.rows], dtype=np.uint64)
        uidx = np.searchsorted(data.user_ids, users)
        hist = np.diff(data.X_binary.indptr)[uidx]
        cold = [int(u) for u, n in zip(users, hist) if n == 0]
        pop = PopularityModel().fit(data, self.cfg, self.cfg["seed"])
        ids = np.array(sorted(self.allowed), dtype=np.int64)
        scores = pop.pop[np.searchsorted(data.item_ids, ids)]
        fallback = topk_items(scores, ids, 5)
        by_user = {int(row["user_id"]): [int(x) for x in row["tv_show_id"].split()]
                   for row in self.rows}
        self.assertEqual(len(cold), self.manifest["recommendation_stats"]["cold_users"])
        for user in cold:
            self.assertEqual(by_user[user], fallback)


if __name__ == "__main__":
    unittest.main()

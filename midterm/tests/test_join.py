import unittest

import numpy as np
import pandas as pd

from midterm.src.data import join_views_to_schedule


def brute(views, schedule, min_seconds=0, min_fraction=0.0):
    sched = schedule.loc[schedule["tv_show_id"] != 0].drop_duplicates(
        subset=["channel_id", "start", "end", "duration", "tv_show_id"]
    )
    out = []
    for _, v in views.iterrows():
        for _, s in sched.iterrows():
            if int(v["vsetv_id"]) != int(s["channel_id"]):
                continue
            a = max(v["start"], s["start"])
            b = min(v["stop"], s["end"])
            ov = int((b - a).total_seconds())
            frac = ov / max(int(s["duration"]), 1)
            if ov > 0 and ov >= min_seconds and frac >= min_fraction:
                out.append((int(v["user_id"]), int(s["tv_show_id"]), ov))
    return sorted(out)


class TestJoin(unittest.TestCase):
    def test_matches_bruteforce_on_nested_duplicate_gap(self):
        sched = pd.DataFrame([
            (1, "2020-01-01 10:00", "2020-01-01 10:30", 1800, 101),
            (1, "2020-01-01 10:15", "2020-01-01 10:20", 300, 102),
            (1, "2020-01-01 10:15", "2020-01-01 10:20", 300, 102),
            (1, "2020-01-01 11:00", "2020-01-01 12:00", 3600, 103),
            (2, "2020-01-01 10:00", "2020-01-01 11:00", 3600, 201),
            (2, "2020-01-01 11:00", "2020-01-01 12:00", 3600, 202),
        ], columns=["channel_id", "start", "end", "duration", "tv_show_id"])
        for c in ("start", "end"):
            sched[c] = pd.to_datetime(sched[c])
        views = pd.DataFrame([
            (1, 1, "2020-01-01 09:50", "2020-01-01 10:10"),
            (2, 1, "2020-01-01 10:10", "2020-01-01 10:50"),
            (3, 1, "2020-01-01 11:00", "2020-01-01 11:01"),
            (4, 2, "2020-01-01 10:59", "2020-01-01 11:01"),
            (5, 2, "2020-01-01 12:00", "2020-01-01 12:10"),
        ], columns=["user_id", "vsetv_id", "start", "stop"])
        for c in ("start", "stop"):
            views[c] = pd.to_datetime(views[c])
        got = join_views_to_schedule(views, sched, min_overlap_seconds=0)
        got_t = sorted((int(r.user_id), int(r.item_id), int(r.overlap_seconds)) for _, r in got.iterrows())
        self.assertEqual(got_t, brute(views, sched))

    def test_random_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        rows = []
        for i in range(12):
            start = pd.Timestamp("2020-02-01") + pd.Timedelta(minutes=int(rng.integers(0, 300)))
            dur = int(rng.integers(1, 90)) * 60
            rows.append((i % 3, start, start + pd.Timedelta(seconds=dur), dur, 1000 + i))
        sched = pd.DataFrame(rows, columns=["channel_id", "start", "end", "duration", "tv_show_id"])
        views = []
        for u in range(20):
            start = pd.Timestamp("2020-02-01") + pd.Timedelta(minutes=int(rng.integers(0, 360)))
            dur = int(rng.integers(1, 120)) * 60
            views.append((u, int(rng.integers(0, 3)), start, start + pd.Timedelta(seconds=dur)))
        views = pd.DataFrame(views, columns=["user_id", "vsetv_id", "start", "stop"])
        got = join_views_to_schedule(views, sched, min_overlap_seconds=60)
        got_t = sorted((int(r.user_id), int(r.item_id), int(r.overlap_seconds)) for _, r in got.iterrows())
        self.assertEqual(got_t, brute(views, sched, min_seconds=60))

    def test_long_view_spans_many_airings(self):
        start = pd.Timestamp("2020-03-01")
        rows = [(1, start + pd.Timedelta(minutes=60 * i), start + pd.Timedelta(minutes=60 * (i + 1)),
                 3600, 5000 + i) for i in range(300)]
        sched = pd.DataFrame(rows, columns=["channel_id", "start", "end", "duration", "tv_show_id"])
        views = pd.DataFrame([(1, 1, start, start + pd.Timedelta(minutes=60 * 300))],
                             columns=["user_id", "vsetv_id", "start", "stop"])
        got = join_views_to_schedule(views, sched, min_overlap_seconds=0)
        self.assertEqual(len(got), 300)


if __name__ == "__main__":
    unittest.main()
"""Airing-level reconstruction of the Kaggle Data-page >=80% target rule.

Fragments for the same user, channel and scheduled airing are unioned to avoid
double counting. This fragment rule and ascending-ID tie break are local
choices; neither is specified on the competition page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


KEY = ["user_id", "item_id", "channel_id", "sched_start", "sched_end"]


def qualify_airings(joined: pd.DataFrame, lo: pd.Timestamp,
                    hi: pd.Timestamp, complete_views: bool = True) -> pd.DataFrame:
    """Return one row per airing with union screen time >= 0.8.

    An airing belongs to [lo, hi) when its start is in the window and its end
    is no later than hi. Complete source views are required by default.
    """
    mask = ((joined["sched_start"] >= lo) & (joined["sched_start"] < hi)
            & (joined["sched_end"] <= hi) & (joined["item_id"] != 0)
            & (joined["overlap_end"] <= hi))
    if complete_views:
        mask &= joined["view_end"] <= hi
    x = joined.loc[mask, KEY + ["overlap_start", "overlap_end", "view_start"]].copy()
    if x.empty:
        return pd.DataFrame(columns=KEY + ["watched_seconds", "duration", "screen_time", "fragment_count", "last_view"])
    x.sort_values(KEY + ["overlap_start", "overlap_end"], kind="mergesort", inplace=True)
    # groupby keeps uint64 user IDs exact; DataFrame.shift may coerce them to
    # float64 and merge adjacent large IDs that differ by only one.
    group_start = x.groupby(KEY, sort=False).cumcount().eq(0).to_numpy()
    cumulative_end = x.groupby(KEY, sort=False)["overlap_end"].cummax()
    prev_end = cumulative_end.shift().where(~group_start, x["overlap_start"])
    lower = np.maximum(x["overlap_start"].to_numpy(dtype="datetime64[s]").astype("int64"),
                       prev_end.to_numpy(dtype="datetime64[s]").astype("int64"))
    upper = x["overlap_end"].to_numpy(dtype="datetime64[s]").astype("int64")
    x["new_seconds"] = np.maximum(upper - lower, 0)
    a = x.groupby(KEY, sort=False).agg(
        watched_seconds=("new_seconds", "sum"),
        fragment_count=("new_seconds", "size"),
        last_view=("view_start", "max"),
    ).reset_index()
    a["duration"] = (a["sched_end"] - a["sched_start"]).dt.total_seconds()
    a = a.loc[a["duration"] > 0].copy()
    a["watched_seconds"] = np.minimum(a["watched_seconds"], a["duration"])
    a["screen_time"] = a["watched_seconds"] / a["duration"]
    return a.loc[a["screen_time"] >= 0.8].reset_index(drop=True)


def top5_frequency(qualified: pd.DataFrame, candidates: np.ndarray | None = None) -> tuple[dict[int, set[int]], pd.DataFrame]:
    """Count qualified airings by programme and retain at most five per user."""
    q = qualified
    if candidates is not None:
        q = q.loc[q["item_id"].isin(set(map(int, candidates)))]
    freq = q.groupby(["user_id", "item_id"], sort=False).size().rename("frequency").reset_index()
    freq = freq.sort_values(["user_id", "frequency", "item_id"], ascending=[True, False, True], kind="mergesort")
    top = freq.groupby("user_id", sort=False).head(5)
    gt = {int(user): set(map(int, rows["item_id"])) for user, rows in top.groupby("user_id", sort=False)}
    return gt, top

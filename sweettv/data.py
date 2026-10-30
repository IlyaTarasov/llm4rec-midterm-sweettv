"""Source loading, exact interval joins and qualified training construction."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from .models import TrainData

SCHEDULE_FILES = (
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)

SCHEDULE_DTYPES = {
    "channel_id": "int64",
    "channel_title": "string",
    "start_time": "string",
    "duration": "int64",
    "tv_show_title": "string",
    "tv_show_id": "int64",
    "tv_show_category": "string",
    "tv_show_genre_1": "string",
    "tv_show_genre_2": "string",
    "tv_show_genre_3": "string",
    "year_of_production": "string",
    "director": "string",
    "actors": "string",
}

VIEW_DTYPES = {
    "user_id": "uint64",
    "vsetv_id": "int64",
    "start_time": "string",
    "stop_time": "string",
    "duraton": "int64",
}

def read_schedule(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path), encoding="utf-8", dtype=SCHEDULE_DTYPES, keep_default_na=False)
    df["start"] = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype("int64")
    df["end"] = df["start"] + pd.to_timedelta(df["duration"], unit="s")
    return df

def read_views(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path), encoding="utf-8", dtype=VIEW_DTYPES)
    df["start"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    df["stop"] = pd.to_datetime(df["stop_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    return df

def read_submission(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), encoding="utf-8", dtype={"user_id": "uint64", "tv_show_id": "string"})

def _labelled_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return unique labelled rows, sorted by channel and start.

    The two training exports overlap at the 2020-05-18 boundary.  Exact
    duplicate rows are removed here so a boundary airing is counted once.
    """
    cols = ["channel_id", "start", "end", "duration", "tv_show_id"]
    lab = schedule.loc[schedule["tv_show_id"] != 0, cols].copy()
    lab = lab.drop_duplicates(subset=cols, keep="first")
    return lab.sort_values(["channel_id", "start"], kind="mergesort")

def join_views_to_schedule(
    views: pd.DataFrame,
    schedule: pd.DataFrame,
    min_overlap_seconds: int = 0,
    min_overlap_fraction: float = 0.0,
    best_overlap_only: bool = False,
) -> pd.DataFrame:
    """Join views to labelled schedule rows by exact positive interval overlap.

    For each channel the intervals are sorted by start.  A prefix maximum of
    interval ends makes it possible to skip every interval that cannot overlap:
    ``lo`` is the first index whose prefix-max end is after the view start and
    ``hi`` is the first interval starting at/after the view stop.  Only indices
    in ``[lo, hi)`` can overlap, and each is checked explicitly.  This is
    complete for gaps, nested/overlapping airings, duplicate rows, and long
    views; it does not use arbitrary row offsets.
    """
    lab = _labelled_schedule(schedule)
    sched_groups = {int(ch): g for ch, g in lab.groupby("channel_id", sort=False)}

    pieces: List[pd.DataFrame] = []
    for ch, g in views.groupby("vsetv_id", sort=False):
        s = sched_groups.get(int(ch))
        if s is None or len(s) == 0:
            continue
        ss = s["start"].values.astype("datetime64[s]").astype("int64")
        se = s["end"].values.astype("datetime64[s]").astype("int64")
        sd = s["duration"].values.astype("int64")
        si = s["tv_show_id"].values.astype("int64")

        vs = g["start"].values.astype("datetime64[s]").astype("int64")
        ve = g["stop"].values.astype("datetime64[s]").astype("int64")
        uid = g["user_id"].values.astype("uint64")
        view_end = g["stop"].values

        prefix_max_end = np.maximum.accumulate(se)
        hi = np.searchsorted(ss, ve, side="left")          # start < view stop
        lo = np.searchsorted(prefix_max_end, vs, side="right")  # end > view start
        width = np.maximum(hi - lo, 0)
        max_width = int(width.max()) if len(width) else 0
        # Normal case: a handful of candidates.  Large widths are handled by an
        # exact per-view fallback below rather than an unbounded broadcast.
        limit = min(max_width, 256)
        for k in range(limit):
            # Views wider than the vectorised limit are handled once by the
            # exact fallback below; do not emit their first `limit` overlaps
            # here as well.
            valid = (k < width) & (width <= limit)
            if not valid.any():
                continue
            idx = lo + k
            ii = np.where(valid, idx, 0)
            ov_start = np.maximum(vs, ss[ii])
            ov_end = np.minimum(ve, se[ii])
            ov = ov_end - ov_start
            frac = ov / np.maximum(sd[ii], 1)
            good = (
                valid
                & (ss[ii] < ve)
                & (se[ii] > vs)
                & (ov > 0)
                & (ov >= min_overlap_seconds)
                & (frac >= min_overlap_fraction)
            )
            if good.any():
                gg = np.flatnonzero(good)
                pieces.append(pd.DataFrame({
                    "user_id": uid[gg],
                    "item_id": si[ii[gg]],
                    "channel_id": np.int64(ch),
                    "view_start": g["start"].values[gg],
                    "view_end": view_end[gg],
                    "overlap_start": ov_start[gg].astype("datetime64[s]"),
                    "overlap_end": ov_end[gg].astype("datetime64[s]"),
                    "overlap_seconds": ov[gg].astype("int64"),
                    "overlap_fraction": frac[gg].astype("float64"),
                    "sched_start": s["start"].values[ii[gg]],
                    "sched_end": s["end"].values[ii[gg]],
                }))

        # Exact fallback for unusually long views that span many airings.
        for pos in np.flatnonzero(width > limit):
            for i in range(int(lo[pos]), int(hi[pos])):
                ov = int(min(ve[pos], se[i]) - max(vs[pos], ss[i]))
                if ov <= 0 or ss[i] >= ve[pos] or se[i] <= vs[pos]:
                    continue
                frac = ov / max(int(sd[i]), 1)
                if ov < min_overlap_seconds or frac < min_overlap_fraction:
                    continue
                pieces.append(pd.DataFrame([{
                    "user_id": int(uid[pos]),
                    "item_id": int(si[i]),
                    "channel_id": int(ch),
                    "view_start": g["start"].values[pos],
                    "view_end": view_end[pos],
                    "overlap_start": pd.Timestamp(max(vs[pos], ss[i]), unit="s"),
                    "overlap_end": pd.Timestamp(min(ve[pos], se[i]), unit="s"),
                    "overlap_seconds": int(ov),
                    "overlap_fraction": float(frac),
                    "sched_start": s["start"].values[i],
                    "sched_end": s["end"].values[i],
                }]))

    if not pieces:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "channel_id", "view_start", "view_end",
            "overlap_start", "overlap_end", "overlap_seconds", "overlap_fraction",
            "sched_start", "sched_end",
        ])
    out = pd.concat(pieces, ignore_index=True)
    if best_overlap_only:
        out = out.sort_values(
            ["user_id", "view_start", "overlap_seconds", "item_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).drop_duplicates(["user_id", "view_start"], keep="first")
    return out

def aggregate_interactions(joined: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level overlaps into one row per (user, programme)."""
    if joined.empty:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "view_count", "watched_seconds",
            "max_overlap_fraction", "first_view", "last_view",
        ])
    g = (
        joined.groupby(["user_id", "item_id"], sort=False)
        .agg(
            view_count=("overlap_seconds", "size"),
            watched_seconds=("overlap_seconds", "sum"),
            max_overlap_fraction=("overlap_fraction", "max"),
            first_view=("view_start", "min"),
            last_view=("view_start", "max"),
        )
        .reset_index()
    )
    return g

def candidate_items(schedule: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> np.ndarray:
    mask = (
        (schedule["tv_show_id"] != 0)
        & (schedule["start"] >= lo)
        & (schedule["start"] < hi)
    )
    return np.sort(schedule.loc[mask, "tv_show_id"].astype("int64").unique())

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

def build_train_data(
    cfg: Dict,
    joined: pd.DataFrame,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    item_meta: pd.DataFrame,
    fit_end: pd.Timestamp,
) -> TrainData:
    jc = cfg["join"]
    j = joined.loc[
        (joined["overlap_end"] <= fit_end)
        & (joined["overlap_seconds"] >= int(jc["min_overlap_seconds"]))
        & (joined["overlap_fraction"] >= float(jc["min_overlap_fraction"]))
    ]
    inter = aggregate_interactions(j)
    if inter.empty:
        raise RuntimeError("no training interactions before fit_end")
    uidx = np.searchsorted(user_ids, inter["user_id"].to_numpy(dtype=np.uint64))
    iidx = np.searchsorted(item_ids, inter["item_id"].to_numpy(dtype=np.int64))
    if not np.array_equal(user_ids[uidx], inter["user_id"].to_numpy(dtype=np.uint64)):
        raise RuntimeError("user vocabulary mismatch")
    if not np.array_equal(item_ids[iidx], inter["item_id"].to_numpy(dtype=np.int64)):
        raise RuntimeError("item vocabulary mismatch")
    inter = inter.assign(uidx=uidx, iidx=iidx)
    inter["last_ts"] = inter["last_view"].values.astype("datetime64[s]").astype("int64")
    rows = inter["uidx"].to_numpy()
    cols = inter["iidx"].to_numpy()
    n_users, n_items = len(user_ids), len(item_ids)
    Xb = sparse.csr_matrix((np.ones(len(inter), dtype=float), (rows, cols)), shape=(n_users, n_items))
    Xc = sparse.csr_matrix((inter["view_count"].to_numpy(dtype=float), (rows, cols)), shape=(n_users, n_items))
    Xs = sparse.csr_matrix((inter["watched_seconds"].to_numpy(dtype=float), (rows, cols)), shape=(n_users, n_items))
    Xl = sparse.csr_matrix((inter["last_ts"].to_numpy(dtype=float), (rows, cols)), shape=(n_users, n_items))
    return TrainData(
        user_ids=user_ids,
        item_ids=item_ids,
        interactions=inter,
        item_meta=item_meta,
        n_users=n_users,
        n_items=n_items,
        X_binary=Xb,
        X_count=Xc,
        X_seconds=Xs,
        X_last=Xl,
        window_start_ts=int(fit_end.value // 10**9),
        seed=int(cfg["seed"]),
    )

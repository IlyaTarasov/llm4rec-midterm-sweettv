"""Data loading, interval join, interaction construction and item metadata."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import resolve

SCHEDULE_FILES = (
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
TRAIN_SCHEDULE_FILES = SCHEDULE_FILES[:2]
TEST_SCHEDULE_FILE = SCHEDULE_FILES[2]
VIEW_FILE = "dataset11-30.csv"
SUBMISSION_FILE = "submission.csv"

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
    df = pd.read_csv(resolve(path), encoding="utf-8", dtype=SCHEDULE_DTYPES, keep_default_na=False)
    df["start"] = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S", errors="raise")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype("int64")
    df["end"] = df["start"] + pd.to_timedelta(df["duration"], unit="s")
    return df


def read_views(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve(path), encoding="utf-8", dtype=VIEW_DTYPES)
    df["start"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    df["stop"] = pd.to_datetime(df["stop_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    return df


def read_submission(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve(path), encoding="utf-8", dtype={"user_id": "uint64", "tv_show_id": "string"})


def load_all(cfg: Dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    data_dir = resolve(cfg["data_dir"])
    schedules = {name: read_schedule(data_dir / name) for name in SCHEDULE_FILES}
    views = read_views(data_dir / VIEW_FILE)
    submission = read_submission(data_dir / SUBMISSION_FILE)
    return schedules, views, submission


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


def parse_year(value: str) -> float:
    if not isinstance(value, str):
        return float("nan")
    m = re.search(r"((?:19|20)\d{2})", value)
    return float(m.group(1)) if m else float("nan")


def _first_nonempty(values: Iterable[str]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_item_metadata(schedules: Sequence[pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    """Aggregate programme attributes from schedule rows strictly before cutoff.

    Only the programme/channel/airtime columns are used for candidate
    availability elsewhere.  Genres, cast, director, year, title and category
    here are training-time metadata; no post-cutoff schedule metadata is read.
    """
    cols = [
        "tv_show_id", "start", "duration", "tv_show_title", "tv_show_category",
        "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
        "year_of_production", "director", "actors", "channel_id",
    ]
    frames = []
    for s in schedules:
        x = s.loc[(s["tv_show_id"] != 0) & (s["start"] < cutoff), cols]
        frames.append(x)
    if not frames:
        return pd.DataFrame(columns=["title", "category", "genres", "year", "director",
                                     "actors", "channels", "n_airings"])
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows.drop_duplicates(
        subset=["channel_id", "start", "duration", "tv_show_id"], keep="first"
    )

    recs = []
    for item, g in allrows.groupby("tv_show_id", sort=True):
        title_counts = g["tv_show_title"].value_counts()
        cat_counts = g["tv_show_category"].value_counts()
        genres = set()
        for col in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
            for v in g[col].unique():
                if isinstance(v, str) and v.strip():
                    genres.add(v.strip())
        years = pd.Series([parse_year(v) for v in g["year_of_production"].unique()]).dropna()
        year = float(years.mode().iloc[0]) if len(years) else float("nan")
        recs.append({
            "item_id": int(item),
            "title": title_counts.index[0] if len(title_counts) else "",
            "category": cat_counts.index[0] if len(cat_counts) else "",
            "genres": sorted(genres),
            "year": year,
            "director": _first_nonempty(g["director"].unique()),
            "actors": _first_nonempty(g["actors"].unique()),
            "channels": sorted({int(c) for c in g["channel_id"].unique()}),
            "n_airings": int(len(g)),
        })
    if not recs:
        return pd.DataFrame(columns=["title", "category", "genres", "year", "director",
                                     "actors", "channels", "n_airings"])
    return pd.DataFrame(recs).set_index("item_id").sort_index()


def window_bounds(cfg: Dict) -> Dict[str, pd.Timestamp]:
    w = cfg["windows"]
    return {
        "fit_start": pd.Timestamp(w["fit_start"]),
        "val_start": pd.Timestamp(w["val_start"]),
        "val_end": pd.Timestamp(w["val_end"]),
        "audit_end": pd.Timestamp(w["audit_end"]),
        "test_end": pd.Timestamp(w["test_end"]),
    }


def candidate_items(schedule: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> np.ndarray:
    mask = (
        (schedule["tv_show_id"] != 0)
        & (schedule["start"] >= lo)
        & (schedule["start"] < hi)
    )
    return np.sort(schedule.loc[mask, "tv_show_id"].astype("int64").unique())


def interactions_in_window(interactions: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    mask = (interactions["last_view"] >= lo) & (interactions["last_view"] < hi)
    return interactions.loc[mask].copy()


def metadata_coverage(item_ids: Sequence[int], meta: pd.DataFrame) -> Dict[str, float]:
    ids = pd.Index(item_ids)
    sub = meta.reindex(ids)
    n = max(len(ids), 1)
    return {
        "n_items": int(len(ids)),
        "has_category": float(sub["category"].fillna("").astype(str).str.len().gt(0).sum() / n),
        "has_genre": float(sub["genres"].apply(lambda x: isinstance(x, list) and len(x) > 0).sum() / n),
        "has_year": float(sub["year"].notna().sum() / n),
        "has_director": float(sub["director"].fillna("").astype(str).str.len().gt(0).sum() / n),
        "has_actors": float(sub["actors"].fillna("").astype(str).str.len().gt(0).sum() / n),
    }

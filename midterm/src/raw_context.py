"""Training-only channel/time context from raw view intervals and schedule airings.

This is an auxiliary exposure signal. It does not replace a permitted method.
Unlabelled airtime contributes to the user's raw viewing context, never to an
item label. Candidate features use only ID, channel, start time and duration.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import resolve


RAW_CACHE = resolve("midterm/artifacts/raw_context.npz")
RAW_MANIFEST = resolve("midterm/results/continuation/raw_context_manifest.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_raw_context() -> dict:
    """Read original views once and cache compact columns under ignored artifacts/."""
    t = time.perf_counter()
    source = resolve("midterm/data/sweettv-tv-program-recommender/dataset11-30.csv")
    vocab = np.load(resolve("midterm/artifacts/vocab.npz"))
    user_ids = vocab["user_ids"]
    df = pd.read_csv(source, usecols=["user_id", "vsetv_id", "start_time", "stop_time", "duraton"],
                     dtype={"user_id": "uint64", "vsetv_id": "int32", "start_time": "string",
                            "stop_time": "string", "duraton": "int64"})
    start = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    stop = pd.to_datetime(df["stop_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    start_ts = start.to_numpy(dtype="datetime64[s]").astype("int64")
    stop_ts = stop.to_numpy(dtype="datetime64[s]").astype("int64")
    uid = df["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(user_ids, uid).astype(np.int32)
    if not np.array_equal(user_ids[uidx], uid):
        raise ValueError("raw view user not found in prepared vocabulary")
    source_duration = df["duraton"].to_numpy(dtype=np.int64)
    measured_duration = stop_ts - start_ts
    diff = source_duration - measured_duration
    channels = df["vsetv_id"].to_numpy(dtype=np.int32)
    hour = start.dt.hour.to_numpy(dtype=np.uint8)
    weekday = start.dt.dayofweek.to_numpy(dtype=np.uint8)
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RAW_CACHE, uidx=uidx, channel=channels, start_ts=start_ts,
             stop_ts=stop_ts, hour=hour, weekday=weekday, user_ids=user_ids)
    manifest = {
        "source": str(source), "source_sha256": _sha256(source),
        "cache": str(RAW_CACHE), "cache_sha256": _sha256(RAW_CACHE),
        "n_views": int(len(df)), "n_users": int(len(user_ids)),
        "n_raw_channels": int(len(np.unique(channels))),
        "n_duration_mismatches": int(np.count_nonzero(diff)),
        "duration_difference_counts": {str(int(x)): int(np.count_nonzero(diff == x))
                                       for x in np.unique(diff[diff != 0])},
        "min_start": str(start.min()), "max_start": str(start.max()),
        "max_stop": str(stop.max()),
        "max_raw_channel_id": int(channels.max()),
        "timing_convention": "naive source timestamps; hour/weekday from view start; view end from stop_time",
        "runtime_seconds": round(time.perf_counter() - t, 3),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    RAW_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RAW_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


@dataclass
class ContextProfile:
    channel_affinity: np.ndarray
    hour_affinity: np.ndarray
    weekhour_affinity: np.ndarray
    n_user_views: np.ndarray
    n_training_views: int
    n_channels: int
    alpha: float


@dataclass
class Opportunity:
    channel: sparse.csr_matrix
    hour: sparse.csr_matrix
    weekhour: sparse.csr_matrix
    n_airings: np.ndarray


def _smoothed_positive_log_lift(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    global_counts = counts.sum(axis=0)
    # Unseen buckets, including channels appearing only in a later schedule,
    # receive zero affinity and do not affect training population statistics.
    seen = global_counts > 0
    background = np.zeros_like(global_counts)
    if not np.any(seen):
        return np.zeros_like(counts, dtype=np.float32)
    background[seen] = (global_counts[seen] + 1.0) / (
        global_counts[seen].sum() + np.count_nonzero(seen)
    )
    total = counts.sum(axis=1, keepdims=True)
    probability = (counts + alpha * background[None, :]) / (total + alpha)
    lift = np.zeros_like(counts)
    np.divide(probability, background[None, :], out=lift, where=seen[None, :])
    np.log(lift, out=lift, where=seen[None, :])
    return np.clip(lift, 0.0, 2.0).astype(np.float32)


def fit_context_profile(
    uidx: np.ndarray,
    channel: np.ndarray,
    start_ts: np.ndarray,
    stop_ts: np.ndarray,
    hour: np.ndarray,
    weekday: np.ndarray,
    user_ids: np.ndarray,
    channel_ids: np.ndarray,
    fit_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    alpha: float = 20.0,
) -> ContextProfile:
    """Fit counts only from complete raw intervals before cutoff."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    lower = int(fit_start.value // 10**9)
    upper = int(cutoff.value // 10**9)
    mask = (start_ts >= lower) & (stop_ts <= upper) & (stop_ts > start_ts)
    u = np.asarray(uidx[mask], dtype=np.int64)
    c = np.asarray(channel[mask], dtype=np.int64)
    h = np.asarray(hour[mask], dtype=np.int64)
    wd = np.asarray(weekday[mask], dtype=np.int64)
    channels = np.asarray(channel_ids, dtype=np.int64)
    cidx = np.searchsorted(channels, c)
    if len(c) and (np.any(cidx >= len(channels)) or not np.array_equal(channels[cidx], c)):
        raise ValueError("channel_ids must contain every raw training channel")
    n_users, n_channels = len(user_ids), len(channels)
    counts_channel = np.bincount(u * n_channels + cidx,
                                 minlength=n_users * n_channels).reshape(n_users, n_channels)
    counts_hour = np.bincount(u * 24 + h, minlength=n_users * 24).reshape(n_users, 24)
    weekhour = wd * 24 + h
    counts_weekhour = np.bincount(u * 168 + weekhour,
                                  minlength=n_users * 168).reshape(n_users, 168)
    return ContextProfile(
        channel_affinity=_smoothed_positive_log_lift(counts_channel, alpha),
        hour_affinity=_smoothed_positive_log_lift(counts_hour, alpha),
        weekhour_affinity=_smoothed_positive_log_lift(counts_weekhour, alpha),
        n_user_views=counts_channel.sum(axis=1).astype(np.int64),
        n_training_views=int(len(u)), n_channels=n_channels, alpha=float(alpha),
    )


def labelled_view_arrays(joined: pd.DataFrame, user_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Ablation: only raw views with a >=60 s labelled overlap, deduplicated."""
    j = joined.loc[joined["overlap_seconds"] >= 60,
                   ["user_id", "channel_id", "view_start", "view_end"]].drop_duplicates()
    uid = j["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(user_ids, uid).astype(np.int32)
    if not np.array_equal(user_ids[uidx], uid):
        raise ValueError("labelled view user not in vocabulary")
    start = j["view_start"]
    stop = j["view_end"]
    return {
        "uidx": uidx,
        "channel": j["channel_id"].to_numpy(dtype=np.int32),
        "start_ts": start.to_numpy(dtype="datetime64[s]").astype("int64"),
        "stop_ts": stop.to_numpy(dtype="datetime64[s]").astype("int64"),
        "hour": start.dt.hour.to_numpy(dtype=np.uint8),
        "weekday": start.dt.dayofweek.to_numpy(dtype=np.uint8),
    }


def build_opportunity(
    schedule: pd.DataFrame,
    candidate_ids: np.ndarray,
    channel_ids: np.ndarray,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> Opportunity:
    """Mean exposure across each item's distinct labelled future airings."""
    a = schedule.loc[
        (schedule["tv_show_id"] != 0) & (schedule["start"] >= lo)
        & (schedule["start"] < hi) & (schedule["duration"] > 0),
        ["tv_show_id", "channel_id", "start", "duration"],
    ].drop_duplicates()
    items = a["tv_show_id"].to_numpy(dtype=np.int64)
    channels = a["channel_id"].to_numpy(dtype=np.int64)
    item_pos = np.searchsorted(candidate_ids, items)
    channel_pos = np.searchsorted(channel_ids, channels)
    if len(items):
        if np.any(item_pos >= len(candidate_ids)) or not np.array_equal(candidate_ids[item_pos], items):
            raise ValueError("candidate_ids missing a schedule item")
        if np.any(channel_pos >= len(channel_ids)) or not np.array_equal(channel_ids[channel_pos], channels):
            raise ValueError("channel_ids missing a schedule channel")
    n_items = len(candidate_ids)
    n_airings = np.bincount(item_pos, minlength=n_items)
    weight = 1.0 / np.maximum(n_airings[item_pos], 1)
    hour = a["start"].dt.hour.to_numpy(dtype=np.int64)
    weekday = a["start"].dt.dayofweek.to_numpy(dtype=np.int64)
    def matrix(col: np.ndarray, n_cols: int) -> sparse.csr_matrix:
        return sparse.csr_matrix((weight, (item_pos, col)), shape=(n_items, n_cols))
    return Opportunity(
        channel=matrix(channel_pos, len(channel_ids)),
        hour=matrix(hour, 24),
        weekhour=matrix(weekday * 24 + hour, 168),
        n_airings=n_airings,
    )


def score_exposure(profile: ContextProfile, opp: Opportunity, uidx: np.ndarray) -> dict[str, np.ndarray]:
    """Channel-only and channel×time variants; every item uses mean airing exposure."""
    def apply(feature: sparse.csr_matrix, affinity: np.ndarray) -> np.ndarray:
        raw = np.asarray(feature.dot(affinity[uidx].T).T, dtype=np.float64)
        max_per_user = raw.max(axis=1, keepdims=True)
        return raw / np.where(max_per_user > 0, max_per_user, 1.0)
    channel = apply(opp.channel, profile.channel_affinity)
    hour = apply(opp.hour, profile.hour_affinity)
    weekhour = apply(opp.weekhour, profile.weekhour_affinity)
    return {
        "channel": channel,
        "channel_hour": channel * (0.75 + 0.25 * hour),
        "channel_weekhour": channel * (0.75 + 0.25 * weekhour),
        "hour_only": hour,
    }


if __name__ == "__main__":
    print(json.dumps(prepare_raw_context(), indent=2))

"""Deterministic ranking, shared fallback and normalized score fusion."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from .models import TrainData

def topk_indices(scores: np.ndarray, k: int = 5, exclude_mask: np.ndarray | None = None) -> np.ndarray:
    """Return k indices ordered by descending score.

    Ties are broken by the position in the candidate array.  All callers pass
    candidate ids sorted ascending, so ties resolve to the smaller programme id.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    if exclude_mask is not None:
        order = order[~exclude_mask[order]]
    return order[:k]

def topk_items(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    k: int = 5,
    exclude_ids: Set[int] | None = None,
) -> List[int]:
    exclude_mask = None
    if exclude_ids:
        exclude_mask = np.fromiter(
            (int(x) in exclude_ids for x in candidate_ids), dtype=bool, count=len(candidate_ids)
        )
    idx = topk_indices(scores, k=k, exclude_mask=exclude_mask)
    return [int(candidate_ids[i]) for i in idx]

def seen_map(data: TrainData) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for u, g in data.interactions.groupby("user_id", sort=False):
        out[int(u)] = set(int(x) for x in g["item_id"].unique())
    return out

def recommend_from_scores(
    scores: np.ndarray,
    users: np.ndarray,
    candidate_ids: np.ndarray,
    data: TrainData,
    repeat_policy: str,
    fallback_scores: np.ndarray,
    k: int = 5,
    cold_mask: np.ndarray | None = None,
) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    uidx_eval = np.searchsorted(data.user_ids, users)
    hist_counts = np.diff(data.X_binary.indptr)[uidx_eval]
    if cold_mask is None:
        cold_mask = hist_counts == 0
    else:
        cold_mask = np.asarray(cold_mask, dtype=bool)
        if len(cold_mask) != len(users):
            raise ValueError("cold_mask must have one value per evaluated user")
    seen = seen_map(data) if repeat_policy == "exclude" else {}
    recs: Dict[int, List[int]] = {}
    stats = {"cold_users": 0, "fallback_used": 0, "short_lists": 0}
    for r, u in enumerate(users):
        u = int(u)
        s = np.asarray(scores[r], dtype=np.float64)
        s = np.where(np.isfinite(s), s, -np.inf)
        seen_u = seen.get(u, set())
        if cold_mask[r]:
            stats["cold_users"] += 1
        if cold_mask[r] or not np.isfinite(s).any() or np.max(s) <= 0:
            s = np.asarray(fallback_scores, dtype=np.float64)
            stats["fallback_used"] += 1
        rec = topk_items(s, candidate_ids, k=k, exclude_ids=seen_u)
        if len(rec) < k:
            fill = np.asarray(fallback_scores, dtype=np.float64).copy()
            fill_rec = topk_items(fill, candidate_ids, k=k - len(rec), exclude_ids=seen_u | set(rec))
            rec.extend(fill_rec)
        if len(rec) < k:
            # Final documented fallback: permit repeats to guarantee five unique ids.
            fill_rec = topk_items(np.asarray(fallback_scores, dtype=np.float64), candidate_ids,
                                  k=k - len(rec), exclude_ids=set(rec))
            rec.extend(fill_rec)
        rec = list(dict.fromkeys(rec))[:k]
        if len(rec) < k:
            stats["short_lists"] += 1
        recs[u] = rec
    return recs, stats

def fuse_scores(
    score_map: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    normalizations: Mapping[str, str],
) -> np.ndarray:
    """Weighted score fusion with explicit per-component normalisation.

    `global` divides by the maximum over all users and candidates (appropriate
    for a user-independent popularity prior).  `user` divides each row by that
    user's maximum, so a component contributes a comparable [0, 1] preference.
    """
    first = next(iter(score_map.values()))
    out = np.zeros_like(first, dtype=np.float64)
    for name, w in weights.items():
        if w == 0 or name not in score_map:
            continue
        s = np.asarray(score_map[name], dtype=np.float64)
        mode = normalizations.get(name, "user")
        if mode == "global":
            m = float(np.max(s)) if s.size else 0.0
            s = s / m if m > 0 else s
        else:
            m = s.max(axis=1, keepdims=True)
            m = np.where(m > 0, m, 1.0)
            s = s / m
        out += float(w) * s
    return out

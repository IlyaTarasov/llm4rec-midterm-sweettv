"""Ranking and MAP@K utilities with deterministic tie handling."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np


def ap_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 5) -> float | None:
    """Average precision at k. Returns None when there is no relevant item."""
    if not relevant:
        return None
    hits = 0
    total = 0.0
    seen = set()
    for rank, item in enumerate(recommended[:k], start=1):
        item = int(item)
        # A repeated id is invalid as a second recommendation.  It consumes the
        # rank but cannot count as another hit; this prevents duplicate
        # predictions from inflating AP.
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


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


def evaluate_recommendations(
    recs: Mapping[int, Sequence[int]],
    gt: Mapping[int, Set[int]],
    users: Sequence[int],
    k: int = 5,
    n_candidates: int | None = None,
) -> Dict[str, float]:
    """MAP@k over eligible users and over the full requested user population.

    `map5_all` assigns zero to a user with no held-out relevant item; this
    mirrors a leaderboard that averages over every submitted user.  `map5_eligible`
    is the conditional metric on users with at least one relevant item and is the
    primary model-selection statistic.
    """
    eligible_aps: List[float] = []
    all_aps: List[float] = []
    recalls: List[float] = []
    hits = 0
    recommended_items: Set[int] = set()
    short_lists = 0
    for u in users:
        u = int(u)
        r = [int(x) for x in recs.get(u, [])]
        g = {int(x) for x in gt.get(u, set())}
        recommended_items.update(set(r))
        if len(r) < k:
            short_lists += 1
        ap = ap_at_k(r, g, k)
        if ap is None:
            all_aps.append(0.0)
            continue
        eligible_aps.append(ap)
        all_aps.append(ap)
        found = len(set(r[:k]) & g)
        recalls.append(found / len(g))
        if found:
            hits += 1
    n = max(len(users), 1)
    return {
        "k": int(k),
        "n_users": int(len(users)),
        "n_eligible": int(len(eligible_aps)),
        "coverage_users": float(len(eligible_aps) / n),
        "map5_eligible": float(np.mean(eligible_aps)) if eligible_aps else float("nan"),
        "map5_all": float(np.mean(all_aps)) if all_aps else float("nan"),
        "recall5_eligible": float(np.mean(recalls)) if recalls else float("nan"),
        "hit_rate_eligible": float(hits / len(eligible_aps)) if eligible_aps else float("nan"),
        "short_lists": int(short_lists),
        "distinct_recommended": int(len(recommended_items)),
        "catalog_coverage": float(len(recommended_items) / n_candidates) if n_candidates else float("nan"),
    }
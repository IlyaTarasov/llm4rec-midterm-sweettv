"""Dataset construction, recommendation assembly, and validation/audit evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .config import resolve
from .data import (
    aggregate_interactions,
    build_item_metadata,
    candidate_items,
    join_views_to_schedule,
    load_all,
    metadata_coverage,
    window_bounds,
)
from .metrics import evaluate_recommendations, topk_items
from .models import (
    AssociationModel,
    BaseModel,
    ContentModel,
    HybridModel,
    MFModel,
    PageRankModel,
    PopularityModel,
    TrainData,
)


def build_vocab(views: pd.DataFrame, schedules: Sequence[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    user_ids = np.sort(views["user_id"].astype("uint64").unique())
    item_parts = [s.loc[s["tv_show_id"] != 0, "tv_show_id"].astype("int64").to_numpy() for s in schedules]
    item_ids = np.sort(np.unique(np.concatenate(item_parts)))
    return user_ids, item_ids


def prepare(cfg: Dict) -> Dict:
    t0 = time.time()
    schedules, views, submission = load_all(cfg)
    train_schedule = pd.concat([schedules["export_arh_11-20-final.csv"],
                                schedules["export_arh_21-30-final.csv"]], ignore_index=True)
    key = ["channel_id", "start", "duration", "tv_show_id"]
    n_before_dedup = len(train_schedule)
    train_schedule = train_schedule.drop_duplicates(subset=key, keep="first")
    n_boundary_dupes = int(n_before_dedup - len(train_schedule))
    # Store every positive overlap once.  Stage-specific watched thresholds are
    # applied in build_train_data/build_eval_data, so sensitivity does not
    # require a new join.
    joined = join_views_to_schedule(
        views, train_schedule,
        min_overlap_seconds=0,
        min_overlap_fraction=0.0,
        best_overlap_only=False,
    )
    user_ids, item_ids = build_vocab(views, list(schedules.values()))
    bounds = window_bounds(cfg)
    manifest = {
        "n_views": int(len(views)),
        "n_joined_rows": int(len(joined)),
        "n_users": int(len(user_ids)),
        "n_items": int(len(item_ids)),
        "n_submission_users": int(len(submission)),
        "view_min": str(views["start"].min()),
        "view_max": str(views["start"].max()),
        "joined_view_min": str(joined["view_start"].min()),
        "joined_view_max": str(joined["view_start"].max()),
        "test_start": str(bounds["audit_end"]),
        "views_on_unmapped_channels": int((~views["vsetv_id"].isin(set(train_schedule["channel_id"]))).sum()),
        "unmapped_channels": sorted({int(x) for x in set(views["vsetv_id"]) - set(train_schedule["channel_id"])}),
        "n_train_schedule_rows": int(len(train_schedule)),
        "n_boundary_duplicate_rows_removed": n_boundary_dupes,
        "n_schedule_rows_after_audit_cutoff": int((train_schedule["start"] >= bounds["audit_end"]).sum()),
        "metadata_policy": "training schedules only, start < stage cutoff; test schedule used only for candidate availability",
        "item_index_policy": "item_ids is the output id namespace; all learned item statistics use training data only",
        "prepared_join": {"min_overlap_seconds": 0, "min_overlap_fraction": 0.0,
                          "best_overlap_only": False, "algorithm": "exact interval overlap"},
        "stage_join_thresholds": cfg["join"],
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    art = resolve(cfg["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    joined.to_pickle(art / "joined_all.pkl")
    np.savez(art / "vocab.npz", user_ids=user_ids, item_ids=item_ids)
    submission.to_pickle(art / "submission.pkl")
    for name, s in schedules.items():
        s.to_pickle(art / f"schedule_{name.replace('.csv','')}.pkl")
    with open(art / "prepare_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def load_prepared(cfg: Dict) -> Dict:
    art = resolve(cfg["artifacts_dir"])
    v = np.load(art / "vocab.npz")
    schedules = {}
    for name in ("export_arh_11-20-final", "export_arh_21-30-final", "export_arh_31-42-final"):
        schedules[name + ".csv"] = pd.read_pickle(art / f"schedule_{name}.pkl")
    return {
        "joined": pd.read_pickle(art / "joined_all.pkl"),
        "user_ids": v["user_ids"],
        "item_ids": v["item_ids"],
        "submission": pd.read_pickle(art / "submission.pkl"),
        "schedules": schedules,
    }


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


def build_eval_data(
    cfg: Dict,
    joined: pd.DataFrame,
    schedule_eval: pd.DataFrame,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    item_meta: pd.DataFrame,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> Dict:
    jc = cfg["join"]
    j = joined.loc[
        (joined["overlap_start"] >= lo)
        & (joined["overlap_end"] <= hi)
        & (joined["overlap_seconds"] >= int(jc["min_overlap_seconds"]))
        & (joined["overlap_fraction"] >= float(jc["min_overlap_fraction"]))
    ]
    inter = aggregate_interactions(j)
    candidates = candidate_items(schedule_eval, lo, hi)
    cand_idx = np.searchsorted(item_ids, candidates)
    if not np.array_equal(item_ids[cand_idx], candidates):
        raise RuntimeError("candidate vocabulary mismatch")
    gt_inter = inter.loc[inter["item_id"].isin(set(int(x) for x in candidates))]
    gt: Dict[int, Set[int]] = {}
    if not gt_inter.empty:
        for u, g in gt_inter.groupby("user_id", sort=False):
            gt[int(u)] = set(int(x) for x in g["item_id"].unique())
    return {
        "lo": lo,
        "hi": hi,
        "candidate_ids": candidates,
        "cand_idx": cand_idx,
        "gt": gt,
        "eval_interactions": gt_inter,
        "n_eval_pairs": int(len(gt_inter)),
        "n_users_with_gt": int(len(gt)),
        "n_candidates": int(len(candidates)),
        "metadata_coverage": metadata_coverage(candidates, item_meta),
    }


def build_candidate_data(
    schedule_eval: pd.DataFrame,
    item_ids: np.ndarray,
    item_meta: pd.DataFrame,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> Dict:
    """Candidate pool only; deliberately does not read held-out interactions."""
    candidates = candidate_items(schedule_eval, lo, hi)
    cand_idx = np.searchsorted(item_ids, candidates)
    if not np.array_equal(item_ids[cand_idx], candidates):
        raise RuntimeError("candidate vocabulary mismatch")
    return {
        "lo": lo,
        "hi": hi,
        "candidate_ids": candidates,
        "cand_idx": cand_idx,
        "n_candidates": int(len(candidates)),
        "metadata_coverage": metadata_coverage(candidates, item_meta),
    }


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


def evaluate_scores(
    scores: np.ndarray,
    users: np.ndarray,
    eval_data: Dict,
    data: TrainData,
    repeat_policy: str,
    fallback_scores: np.ndarray,
    k: int = 5,
    cold_mask: np.ndarray | None = None,
) -> Dict:
    recs, stats = recommend_from_scores(
        scores, users, eval_data["candidate_ids"], data, repeat_policy, fallback_scores,
        k=k, cold_mask=cold_mask,
    )
    metrics = evaluate_recommendations(
        recs, eval_data["gt"], users, k=k, n_candidates=eval_data["n_candidates"]
    )
    metrics.update(stats)
    return {"metrics": metrics, "recs": recs}


def fit_and_score(model: BaseModel, data: TrainData, eval_data: Dict, users: np.ndarray,
                  cfg: Dict, seed: int) -> Tuple[np.ndarray, float]:
    t0 = time.time()
    model.fit(data, cfg, seed)
    uidx_eval = np.searchsorted(data.user_ids, users)
    scores = model.score(uidx_eval, eval_data["cand_idx"], data)
    return scores, time.time() - t0


def rrf_from_scores(score_map: Mapping[str, np.ndarray], weights: Mapping[str, float], rrf_k: int = 60) -> np.ndarray:
    first = next(iter(score_map.values()))
    out = np.zeros_like(first, dtype=np.float64)
    n_cand = first.shape[1]
    base = np.arange(1, n_cand + 1, dtype=np.float64)[None, :]
    for name, w in weights.items():
        if w == 0 or name not in score_map:
            continue
        s = score_map[name]
        order = np.argsort(-s, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(base, order.shape), axis=1)
        out += float(w) / (rrf_k + ranks)
    return out


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


def evaluate_individual(
    models: Mapping[str, BaseModel],
    data: TrainData,
    eval_data: Dict,
    users: np.ndarray,
    cfg: Dict,
    repeat_policy: str,
    fallback_scores: np.ndarray,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    results: Dict[str, Dict] = {}
    score_map: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        scores, elapsed = fit_and_score(model, data, eval_data, users, cfg, int(cfg["seed"]))
        score_map[name] = scores
        res = evaluate_scores(scores, users, eval_data, data, repeat_policy, fallback_scores)
        res["runtime_seconds"] = round(elapsed, 3)
        results[name] = res
    return results, score_map


def summarize_metrics(results: Mapping[str, Dict], users: np.ndarray, eval_data: Dict,
                      data: TrainData) -> pd.DataFrame:
    uidx_eval = np.searchsorted(data.user_ids, users)
    hist_counts = np.diff(data.X_binary.indptr)[uidx_eval]
    cold_users = users[hist_counts == 0]
    warm_users = users[hist_counts > 0]
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        recs = res["recs"]
        row = {"method": name, **m, "runtime_seconds": res.get("runtime_seconds", np.nan)}
        for subset_name, subset in (("cold", cold_users), ("warm", warm_users)):
            if len(subset) == 0:
                continue
            sub = evaluate_recommendations(recs, eval_data["gt"], subset, k=5,
                                           n_candidates=eval_data["n_candidates"])
            row[f"{subset_name}_n_users"] = int(len(subset))
            row[f"{subset_name}_n_eligible"] = sub["n_eligible"]
            row[f"{subset_name}_map5_eligible"] = sub["map5_eligible"]
            row[f"{subset_name}_map5_all"] = sub["map5_all"]
        rows.append(row)
    return pd.DataFrame(rows)

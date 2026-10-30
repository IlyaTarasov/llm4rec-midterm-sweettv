"""Temporal evaluation against reconstructed >=80% top-five-frequency targets."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .context_experiment import _empty_metadata, _schedule_for
from .data import candidate_items
from .evaluate import build_train_data, fuse_scores, load_prepared, recommend_from_scores
from .metrics import evaluate_recommendations
from .models import AssociationModel, PopularityModel, RepeatModel
from .pipeline import cfg_with, sha256_file
from .qualified_airings import qualify_airings, top5_frequency

ROOT = resolve("midterm/results/continuation")
PLAN = ROOT / "corrected_plan.json"


def _qualified_training(q, cfg, p, cutoff):
    pseudo = q[["user_id", "item_id", "watched_seconds", "screen_time", "last_view", "sched_end"]].copy()
    pseudo.rename(columns={"watched_seconds": "overlap_seconds", "screen_time": "overlap_fraction",
                           "last_view": "view_start", "sched_end": "overlap_end"}, inplace=True)
    loose = cfg_with(cfg, {"join": {"min_overlap_seconds": 0, "min_overlap_fraction": 0.0}})
    return build_train_data(loose, pseudo, p["user_ids"], p["item_ids"], _empty_metadata(), cutoff)


def _scores(train, cfg, users, cand_idx, population):
    uidx = np.searchsorted(train.user_ids, users)
    pop = PopularityModel().fit(train, cfg_with(cfg, {"popularity": {"weight": "users"}}), cfg["seed"]).score(uidx, cand_idx, train)
    repeat = RepeatModel().fit(train, cfg_with(cfg, {"repeat": {"half_life_days": 14.0}}), cfg["seed"]).score(uidx, cand_idx, train)
    assoc_cfg = cfg_with(cfg, {"association": {"min_support": 10, "top_k": 100,
                                               "half_life_days": 30.0, "lift_population": population}})
    assoc = AssociationModel().fit(train, assoc_cfg, cfg["seed"]).score(uidx, cand_idx, train)
    return pop, repeat, assoc


def _fuse(pop, repeat, assoc, repeat_weight):
    return fuse_scores({"popularity": pop, "repeat": repeat, "association": assoc},
                       {"popularity": 1.0, "repeat": repeat_weight, "association": 0.25},
                       {"popularity": "global", "repeat": "user", "association": "user"})


def evaluate_window(w, cfg, p):
    t = time.perf_counter()
    fit_start, cutoff, end = map(pd.Timestamp, (w["fit_start"], w["cutoff"], w["end"]))
    schedule = _schedule_for(w, p)
    candidates = candidate_items(schedule, cutoff, end)
    cand_idx = np.searchsorted(p["item_ids"], candidates)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    qtrain = qualify_airings(p["joined"], fit_start, cutoff)
    qeval = qualify_airings(p["joined"], cutoff, end)
    gt, ranked = top5_frequency(qeval, candidates)
    eligible5 = np.array([u for u in users if len(gt.get(int(u), ())) == 5], dtype=np.uint64)
    j = p["joined"]
    one = j.loc[(j["sched_start"] >= cutoff) & (j["sched_start"] < end)
                & (j["sched_end"] <= end) & (j["view_end"] <= end)
                & (j["overlap_fraction"] >= 0.8),
                ["user_id", "item_id", "channel_id", "sched_start", "sched_end"]].drop_duplicates()
    gt_single, _ = top5_frequency(one, candidates)
    single_eligible5 = np.array([u for u in users if len(gt_single.get(int(u), ())) == 5], dtype=np.uint64)
    old = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], _empty_metadata(), cutoff)
    qualified = _qualified_training(qtrain, cfg, p, cutoff)
    opop, orepeat, oassoc_legacy = _scores(old, cfg, users, cand_idx, "all_indexed_legacy")
    _, _, oassoc_active = _scores(old, cfg, users, cand_idx, "training_active")
    qpop, qrepeat, qassoc = _scores(qualified, cfg, users, cand_idx, "all_indexed_legacy")
    _, _, qassoc_active = _scores(qualified, cfg, users, cand_idx, "training_active")
    uidx = np.searchsorted(p["user_ids"], users)
    qualified_cold = np.diff(qualified.X_binary.indptr)[uidx] == 0
    variants = {
        "popularity_all60": opop,
        "first_submitted": _fuse(opop, orepeat, oassoc_legacy, 3.0),
        "second_submitted": _fuse(opop, orepeat, oassoc_active, 1.0),
        "first_fair": _fuse(opop, orepeat, oassoc_legacy, 3.0),
        "qualified_popularity": qpop,
        "qualified_repeat": qrepeat,
        "qualified_association": qassoc,
        "qualified_fusion": _fuse(qpop, qrepeat, qassoc, 3.0),
        "qualified_fusion_active": _fuse(qpop, qrepeat, qassoc_active, 3.0),
    }
    fallback = opop[0]
    rows = []
    for name, scores in variants.items():
        cold = None if name in ("first_submitted", "second_submitted", "popularity_all60") else qualified_cold
        recs, stats = recommend_from_scores(scores, users, candidates, old, "allow", fallback, cold_mask=cold)
        allm = evaluate_recommendations(recs, gt, users, k=5, n_candidates=len(candidates))
        fivem = evaluate_recommendations(recs, gt, eligible5, k=5, n_candidates=len(candidates))
        single_all = evaluate_recommendations(recs, gt_single, users, k=5, n_candidates=len(candidates))
        single_five = evaluate_recommendations(recs, gt_single, single_eligible5, k=5, n_candidates=len(candidates))
        rows.append({"window": w["name"], "variant": name, "map5_all_users": allm["map5_all"],
                     "map5_eligible5": fivem["map5_all"], "n_eligible5": len(eligible5),
                     "map5_single_all_users": single_all["map5_all"],
                     "map5_single_eligible5": single_five["map5_all"],
                     "n_single_eligible5": len(single_eligible5),
                     "n_users": len(users), "n_candidates": len(candidates),
                     "fallback_used": stats["fallback_used"]})
    # A single long viewing fragment may itself clear 80%; compare with union.
    union_keys = qeval[["user_id", "item_id", "channel_id", "sched_start", "sched_end"]].drop_duplicates()
    one_keys = one.set_index(list(one.columns)).index
    union_index = union_keys.set_index(list(union_keys.columns)).index
    stats = {"window": w["name"], "qualified_train_airings": len(qtrain),
             "qualified_eval_airings": len(qeval), "single_view_qualified_eval_airings": len(one),
             "union_only_airings": int((~union_index.isin(one_keys)).sum()),
             "n_eligible5": len(eligible5), "n_target_users": len(gt),
             "n_qualified_cold_template_users": int(qualified_cold.sum()),
             "n_target_users_with_fewer_than5": sum(0 < len(gt.get(int(u), ())) < 5 for u in users),
             "runtime_seconds": round(time.perf_counter()-t, 2)}
    return rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "diagnostic"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if args.phase == "diagnostic":
        decision = json.loads((ROOT / "corrected_decision.json").read_text(encoding="utf-8"))
        if decision["selection_sha256"] != sha256_file(ROOT / "corrected_selection.csv"):
            raise RuntimeError("selection result changed since decision")
        windows = [plan["diagnostic_window"]]
    else:
        windows = plan["selection_windows"]
    cfg = load_config()
    p = load_prepared(cfg)
    rows, stats = [], []
    for w in windows:
        r, s = evaluate_window(w, cfg, p)
        rows.extend(r); stats.append(s)
        print(w["name"], {x["variant"]: round(x["map5_eligible5"], 5) for x in r}, flush=True)
        print(s, flush=True)
    stem = "corrected_selection" if args.phase == "selection" else "corrected_diagnostic"
    pd.DataFrame(rows).to_csv(ROOT / f"{stem}.csv", index=False)
    (ROOT / f"{stem}_stats.json").write_text(json.dumps({"plan_sha256": sha256_file(PLAN), "windows": stats}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

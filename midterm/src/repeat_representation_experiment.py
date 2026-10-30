"""Fixed repeat-strength ablation and frozen-method proxy diagnostics."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .context_experiment import _all_proxies, _empty_metadata, _schedule_for
from .data import build_item_metadata
from .evaluate import build_train_data, fuse_scores, load_prepared, recommend_from_scores
from .metrics import evaluate_recommendations
from .models import AssociationModel, ContentModel, MFModel, PageRankModel, PopularityModel, RepeatModel
from .pipeline import cfg_for_method, cfg_with, sha256_file
from .raw_context import RAW_CACHE

ROOT = resolve("midterm/results/continuation")
PLAN = ROOT / "repeat_plan.json"
NAMES = ("repeat_capped_minutes", "repeat_distinct_days", "repeat_distinct_weeks")


def repeat_strengths(joined: pd.DataFrame, fit_start: pd.Timestamp,
                     cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per user/item, formed only from complete pre-cutoff views."""
    j = joined.loc[
        (joined["view_start"] >= fit_start) & (joined["view_end"] <= cutoff)
        & (joined["overlap_end"] <= cutoff) & (joined["overlap_seconds"] >= 60)
        & (joined["overlap_fraction"] >= 0),
        ["user_id", "item_id", "view_start", "overlap_seconds"],
    ].copy()
    j["day"] = j["view_start"].dt.floor("D")
    d = j.groupby(["user_id", "item_id", "day"], sort=False).agg(
        seconds=("overlap_seconds", "sum"), last_view=("view_start", "max")
    ).reset_index()
    d["capped_seconds"] = d["seconds"].clip(upper=3600)
    d["week"] = d["day"].dt.to_period("W-SUN")
    return d.groupby(["user_id", "item_id"], sort=False).agg(
        capped_seconds=("capped_seconds", "sum"),
        days=("day", "size"), weeks=("week", "nunique"),
        last_view=("last_view", "max"),
    ).reset_index()


def strength_scores(agg: pd.DataFrame, users: np.ndarray,
                    candidates: np.ndarray, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
    uid = agg["user_id"].to_numpy(dtype=np.uint64)
    iid = agg["item_id"].to_numpy(dtype=np.int64)
    order = np.argsort(users)
    sorted_users = users[order]
    upos = np.searchsorted(sorted_users, uid)
    ipos = np.searchsorted(candidates, iid)
    valid = (upos < len(users)) & (ipos < len(candidates))
    valid &= sorted_users[np.clip(upos, 0, len(users)-1)] == uid
    valid &= candidates[np.clip(ipos, 0, len(candidates)-1)] == iid
    row, col = order[upos[valid]], ipos[valid]
    age = (cutoff - agg.loc[valid, "last_view"]).dt.total_seconds().to_numpy() / 86400.0
    decay = np.power(0.5, np.maximum(age, 0) / 14.0)
    out = {}
    for name, values in (
        ("repeat_capped_minutes", agg["capped_seconds"].to_numpy(dtype=float) / 60.0),
        ("repeat_distinct_days", agg["days"].to_numpy(dtype=float)),
        ("repeat_distinct_weeks", agg["weeks"].to_numpy(dtype=float)),
    ):
        score = np.zeros((len(users), len(candidates)), dtype=np.float64)
        score[row, col] = np.log1p(values[valid]) * decay
        out[name] = score
    return out


def _load_inputs():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    frozen = {"context_selection.csv": plan["frozen_context_selection_sha256"],
              "context_diagnostic.csv": plan["frozen_context_diagnostic_sha256"]}
    for filename, digest in frozen.items():
        if sha256_file(ROOT / filename) != digest:
            raise RuntimeError(f"frozen {filename} changed")
    if sha256_file(resolve("midterm/results/selected_model.json")) != plan["frozen_selected_model_sha256"]:
        raise RuntimeError("selected_model.json changed")
    cfg = load_config()
    p = load_prepared(cfg)
    raw = np.load(RAW_CACHE)
    return plan, cfg, p, raw


def _window_common(window, cfg, p, raw):
    start, cutoff, end = map(pd.Timestamp, (window["fit_start"], window["cutoff"], window["end"]))
    schedule = _schedule_for(window, p)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(p["user_ids"], users)
    train = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], _empty_metadata(), cutoff)
    ev, gt = _all_proxies(cfg, p, schedule, cutoff, end)
    pop = PopularityModel().fit(train, cfg_with(cfg, {"popularity": {"weight": "users"}}), cfg["seed"]).score(uidx, ev["cand_idx"], train)
    mask = (raw["start_ts"] >= int(start.value // 10**9)) & (raw["stop_ts"] <= int(cutoff.value // 10**9)) & (raw["stop_ts"] > raw["start_ts"])
    raw_counts = np.bincount(raw["uidx"][mask], minlength=len(p["user_ids"]))
    cold = raw_counts[uidx] == 0
    return start, cutoff, end, users, uidx, train, ev, gt, pop, cold


def _rows(recs, stats, window, name, users, ev, gt):
    return [{"window": window["name"], "proxy": proxy, "variant": name,
             "n_candidates": ev["n_candidates"], "raw_cold_users": stats["cold_users"],
             "fallback_used": stats["fallback_used"],
             **evaluate_recommendations(recs, target, users, k=5, n_candidates=ev["n_candidates"])}
            for proxy, target in gt.items()]


def repeat_phase(phase: str, plan, cfg, p, raw):
    context_plan = json.loads((ROOT / "context_plan.json").read_text(encoding="utf-8"))
    if phase == "selection":
        windows = context_plan["selection_windows"]
    else:
        decision = json.loads((ROOT / "repeat_decision.json").read_text(encoding="utf-8"))
        if decision["repeat_selection_sha256"] != sha256_file(ROOT / "repeat_selection.csv"):
            raise RuntimeError("repeat selection changed after decision")
        windows = [context_plan["final_diagnostic_window"]]
    rows, window_stats = [], []
    for w in windows:
        t = time.perf_counter()
        start, cutoff, _, users, uidx, train, ev, gt, pop, cold = _window_common(w, cfg, p, raw)
        assoc_cfg = cfg_with(cfg, {"association": {"min_support": 10, "top_k": 100,
                                                  "half_life_days": 30.0,
                                                  "lift_population": "all_indexed_legacy"}})
        assoc = AssociationModel().fit(train, assoc_cfg, cfg["seed"]).score(uidx, ev["cand_idx"], train)
        baseline_repeat = RepeatModel().fit(train, cfg_with(cfg, {"repeat": {"half_life_days": 14.0}}), cfg["seed"]).score(uidx, ev["cand_idx"], train)
        norm = {"popularity": "global", "association": "user", "repeat": "user"}
        def fused(repeat):
            return fuse_scores({"popularity": pop, "association": assoc, "repeat": repeat},
                               {"popularity": 1.0, "association": 0.25, "repeat": 3.0}, norm)
        baseline_recs, baseline_stats = recommend_from_scores(fused(baseline_repeat), users, ev["candidate_ids"], train, "allow", pop[0], cold_mask=cold)
        baseline_rows = _rows(baseline_recs, baseline_stats, w, "first_legacy_repeat3", users, ev, gt)
        frozen_name = "context_selection.csv" if phase == "selection" else "context_diagnostic.csv"
        frozen = pd.read_csv(ROOT / frozen_name)
        frozen = frozen.loc[(frozen["window"] == w["name"]) & (frozen["variant"] == "first_legacy_repeat3")]
        for row in baseline_rows:
            cached = frozen.loc[frozen["proxy"] == row["proxy"], "map5_all"]
            if len(cached) != 1 or abs(float(cached.iloc[0]) - row["map5_all"]) > 1e-12:
                raise RuntimeError("first strategy does not reproduce frozen proxy score")
        agg = repeat_strengths(p["joined"], start, cutoff)
        alternatives = strength_scores(agg, users, ev["candidate_ids"], cutoff)
        if phase == "diagnostic":
            selected = json.loads((ROOT / "repeat_decision.json").read_text(encoding="utf-8"))["diagnostic_probe"]
            alternatives = {selected: alternatives[selected]} if selected else {}
        for name, strength in alternatives.items():
            recs, stats = recommend_from_scores(fused(strength), users, ev["candidate_ids"], train, "allow", pop[0], cold_mask=cold)
            rows += _rows(recs, stats, w, name, users, ev, gt)
        window_stats.append({"window": w["name"], "complete_training_pairs": len(agg),
                             "runtime_seconds": round(time.perf_counter()-t, 3)})
        print(w["name"], {r["variant"]: round(r["map5_all"], 5) for r in rows if r["window"] == w["name"] and r["proxy"] == "all_60"}, flush=True)
    name = "repeat_selection.csv" if phase == "selection" else "repeat_diagnostic.csv"
    pd.DataFrame(rows).to_csv(ROOT / name, index=False)
    if phase == "selection":
        frozen = pd.read_csv(ROOT / "context_selection.csv")
        frozen = frozen.loc[frozen["variant"].isin(plan["comparison_variants"][:4])]
        pd.concat([frozen, pd.DataFrame(rows)], ignore_index=True).to_csv(ROOT / "repeat_comparison.csv", index=False)
    (ROOT / name.replace(".csv", "_stats.json")).write_text(json.dumps({
        "phase": phase, "plan_sha256": sha256_file(PLAN), "windows": window_stats,
        "seed": cfg["seed"]}, indent=2), encoding="utf-8")


def methods_phase(plan, cfg, p, raw):
    cp = json.loads((ROOT / "context_plan.json").read_text(encoding="utf-8"))
    windows = [w for w in cp["selection_windows"] if w["name"] in ("early_35d", "later_35d")]
    selected = json.loads(resolve("midterm/results/selected_model.json").read_text(encoding="utf-8"))
    rows, runtimes = [], []
    for w in windows:
        start, cutoff, _, users, uidx, _, ev, gt, pop, cold = _window_common(w, cfg, p, raw)
        schedule_sources = [p["schedules"]["export_arh_11-20-final.csv"]]
        if w["name"] == "later_35d":
            schedule_sources.append(p["schedules"]["export_arh_21-30-final.csv"])
        t = time.perf_counter()
        meta = build_item_metadata(schedule_sources, cutoff)
        train = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], meta, cutoff)
        for name, cls in (("content", ContentModel), ("mf", MFModel), ("pagerank", PageRankModel)):
            mstart = time.perf_counter()
            model = cls().fit(train, cfg_for_method(cfg, name, selected["method_configs"][name]), cfg["seed"])
            scores = model.score(uidx, ev["cand_idx"], train)
            recs, stats = recommend_from_scores(scores, users, ev["candidate_ids"], train, "allow", pop[0], cold_mask=cold)
            rows += _rows(recs, stats, w, name, users, ev, gt)
            runtimes.append({"window": w["name"], "model": name, "runtime_seconds": round(time.perf_counter()-mstart, 3)})
            print(w["name"], name, round(rows[-5]["map5_all"], 5), flush=True)
        print(w["name"], "elapsed", round(time.perf_counter()-t, 1), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "method_proxy_diagnostics.csv", index=False)
    (ROOT / "method_proxy_stats.json").write_text(json.dumps({"plan_sha256": sha256_file(PLAN), "runtimes": runtimes, "seed": cfg["seed"]}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "diagnostic", "methods"), required=True)
    args = parser.parse_args()
    plan, cfg, p, raw = _load_inputs()
    if args.phase == "methods":
        methods_phase(plan, cfg, p, raw)
    else:
        repeat_phase(args.phase, plan, cfg, p, raw)


if __name__ == "__main__":
    main()

"""Airing-exposure rate and integrated-recency ablations; no submission output."""
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
from .context_experiment import _schedule_for
from .corrected_target_experiment import ROOT
from .evaluate import fuse_scores, load_prepared
from .fourth_pass import evaluate_variant, fused, prepare_window
from .pipeline import sha256_file
from .qualified_airings import qualify_airings

PLAN = ROOT / "fifth_pass_plan.json"
H = (14.0, 28.0, 56.0)


def _airing_rows(schedule, lo, hi):
    s = schedule.copy()
    if "end" not in s:
        s["end"] = s["start"] + pd.to_timedelta(s["duration"], unit="s")
    return s.loc[(s["tv_show_id"] != 0) & (s["start"] >= lo)
                 & (s["start"] < hi) & (s["end"] <= hi),
                 ["tv_show_id", "channel_id", "start", "duration", "end"]].drop_duplicates()


def component_scores(p, w, x):
    start, cutoff, end = map(pd.Timestamp, (w["fit_start"], w["cutoff"], w["end"]))
    candidates = x["candidates"]
    users = x["users"]
    past_schedule = pd.concat([p["schedules"]["export_arh_11-20-final.csv"],
                               p["schedules"]["export_arh_21-30-final.csv"]], ignore_index=True)
    past = _airing_rows(past_schedule, start, cutoff)
    future = _airing_rows(_schedule_for(w, p), cutoff, end)
    future_count = future.groupby("tv_show_id").size().reindex(candidates, fill_value=0).to_numpy(dtype=float)
    future_count = np.minimum(future_count, 2.0 * ((end - cutoff).total_seconds() / 86400.0))
    q = qualify_airings(p["joined"], start, cutoff)
    uid = q["user_id"].to_numpy(dtype=np.uint64)
    iid = q["item_id"].to_numpy(dtype=np.int64)
    order = np.argsort(users)
    sorted_users = users[order]
    upos = np.searchsorted(sorted_users, uid)
    ipos = np.searchsorted(candidates, iid)
    valid = (upos < len(users)) & (ipos < len(candidates))
    valid &= sorted_users[np.clip(upos, 0, len(users)-1)] == uid
    valid &= candidates[np.clip(ipos, 0, len(candidates)-1)] == iid
    row = order[upos[valid]]
    col = ipos[valid]
    q_end = q.loc[valid, "sched_end"]
    past_idx = np.searchsorted(candidates, past["tv_show_id"].to_numpy(dtype=np.int64))
    past_valid = past_idx < len(candidates)
    past_valid &= candidates[np.clip(past_idx, 0, len(candidates)-1)] == past["tv_show_id"].to_numpy(dtype=np.int64)
    out = {}
    for h in H:
        q_age = (cutoff - q_end).dt.total_seconds().to_numpy() / 86400.0
        q_weight = np.power(0.5, np.maximum(q_age, 0) / h)
        count = np.zeros((len(users), len(candidates)), dtype=np.float64)
        np.add.at(count, (row, col), q_weight)
        p_age = (cutoff - past.loc[past_valid, "end"]).dt.total_seconds().to_numpy() / 86400.0
        p_weight = np.power(0.5, np.maximum(p_age, 0) / h)
        exposure = np.bincount(past_idx[past_valid], weights=p_weight, minlength=len(candidates))
        out[f"decayed_h{int(h)}"] = np.log1p(count) * x["opportunity"][None, :]
        for alpha in (1.0, 5.0):
            out[f"rate_h{int(h)}_a{int(alpha)}"] = np.log1p(
                future_count[None, :] * count / (exposure[None, :] + alpha))
    return out, {"n_qualified_train_airings": len(q), "n_past_schedule_airings": len(past),
                 "n_future_schedule_airings": len(future)}


def fuse_component(x, component):
    return fuse_scores({"popularity": x["pop"], "association": x["assoc"], "repeat": component},
                       {"popularity": 1.0, "association": 0.25, "repeat": 2.0},
                       {"popularity": "global", "association": "user", "repeat": "user"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "diagnostic"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if sha256_file(resolve("midterm/submission_candidate_opportunity.csv")) != plan["best_confirmed"]["sha256"]:
        raise RuntimeError("submitted fourth CSV changed")
    if args.phase == "diagnostic":
        decision = json.loads((ROOT / "fifth_selection_decision.json").read_text(encoding="utf-8"))
        if decision["selection_sha256"] != sha256_file(ROOT / "fifth_selection.json"):
            raise RuntimeError("selection changed after decision")
    cp = json.loads((ROOT / "corrected_plan.json").read_text(encoding="utf-8"))
    windows = cp["selection_windows"] if args.phase == "selection" else [cp["diagnostic_window"]]
    cfg = load_config()
    p = load_prepared(cfg)
    rows, window_stats = [], []
    for w in windows:
        t = time.perf_counter()
        x = prepare_window(p, cfg, w)
        baseline, baseline_ap = evaluate_variant(x, "fourth_submitted", fused(x, 28.0, 2.0, 0.25, opportunity=True))
        prior = pd.read_csv(ROOT / ("fourth_ablation.csv" if args.phase == "selection" else "fourth_diagnostic.csv"))
        prior = prior.loc[(prior["window"] == w["name"]) & (prior["variant"] == "opportunity")]
        if len(prior) != 1 or abs(float(prior.iloc[0]["map5_eligible5"]) - baseline["map5_eligible5"]) > 1e-12:
            raise RuntimeError("fourth baseline does not reproduce")
        rows.append(baseline)
        components, stats = component_scores(p, w, x)
        names = list(components) if args.phase == "selection" else [decision["selected_variant"]]
        for name in names:
            result, _ = evaluate_variant(x, name, fuse_component(x, components[name]), baseline_ap)
            rows.append(result)
        window_stats.append({"window": w["name"], **stats, "elapsed_seconds": round(time.perf_counter()-t, 3)})
        print(w["name"], {r["variant"]: round(r["map5_eligible5"], 5) for r in rows if r["window"] == w["name"]}, flush=True)
    output = ROOT / ("fifth_selection.json" if args.phase == "selection" else "fifth_diagnostic.json")
    output.write_text(json.dumps({"plan_sha256": sha256_file(PLAN), "rows": rows,
                                  "window_stats": window_stats, "seed": cfg["seed"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

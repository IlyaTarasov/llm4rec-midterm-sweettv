"""Bounded raw-channel/time exposure experiment and target-proxy diagnostics.

Run selection first, inspect the result, write a selection_decision.json,
then run the diagnostic phase. Both submitted CSVs are read-only here.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .config import load_config, resolve
from .evaluate import build_eval_data, build_train_data, fuse_scores, load_prepared, recommend_from_scores
from .metrics import evaluate_recommendations
from .models import AssociationModel, PopularityModel, RepeatModel
from .pipeline import cfg_with, sha256_file
from .raw_context import (RAW_CACHE, build_opportunity, fit_context_profile,
                          labelled_view_arrays, score_exposure)

ROOT = resolve("midterm/results/continuation")
PLAN_PATH = ROOT / "context_plan.json"


def _empty_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=["category", "genres", "year", "director", "actors"])


def _top5_gt(inter: pd.DataFrame, column: str) -> dict[int, set[int]]:
    ranked = inter.sort_values(["user_id", column, "item_id"],
                               ascending=[True, False, True], kind="mergesort")
    ranked = ranked.groupby("user_id", sort=False).head(5)
    return {int(u): set(map(int, g["item_id"])) for u, g in ranked.groupby("user_id", sort=False)}


def _schedule_for(window: dict, p: dict) -> pd.DataFrame:
    cols = ["tv_show_id", "channel_id", "start", "duration"]
    if window["schedule"] == "31-42":
        return p["schedules"]["export_arh_31-42-final.csv"][cols]
    s0 = p["schedules"]["export_arh_11-20-final.csv"]
    s1 = p["schedules"]["export_arh_21-30-final.csv"]
    if window["schedule"] == "11-20":
        return s0[cols]
    if window["schedule"] == "21-30":
        return s1[cols]
    if window["schedule"] == "11-20+21-30":
        return pd.concat([s0[cols], s1[cols]], ignore_index=True)
    raise ValueError(window["schedule"])


def _all_proxies(cfg: dict, p: dict, schedule: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, dict]:
    meta = _empty_metadata()
    ev = build_eval_data(cfg, p["joined"], schedule, p["user_ids"], p["item_ids"],
                         meta, start, end)
    strict_seconds_cfg = cfg_with(cfg, {"join": {"min_overlap_seconds": 300,
                                                  "min_overlap_fraction": 0.0}})
    strict_fraction_cfg = cfg_with(cfg, {"join": {"min_overlap_seconds": 60,
                                                   "min_overlap_fraction": 0.25}})
    strict_seconds = build_eval_data(strict_seconds_cfg, p["joined"], schedule,
                                     p["user_ids"], p["item_ids"], meta, start, end)
    strict_fraction = build_eval_data(strict_fraction_cfg, p["joined"], schedule,
                                      p["user_ids"], p["item_ids"], meta, start, end)
    assert np.array_equal(ev["candidate_ids"], strict_seconds["candidate_ids"])
    assert np.array_equal(ev["candidate_ids"], strict_fraction["candidate_ids"])
    gt = {
        "all_60": ev["gt"],
        "top5_seconds": _top5_gt(ev["eval_interactions"], "watched_seconds"),
        "top5_count": _top5_gt(ev["eval_interactions"], "view_count"),
        "strict_300": strict_seconds["gt"],
        "strict_fraction25": strict_fraction["gt"],
    }
    return ev, gt


def _make_variants(pop: np.ndarray, association_legacy: np.ndarray,
                   association_active: np.ndarray, repeat: np.ndarray,
                   raw_scores: dict[str, np.ndarray],
                   labelled_channel: np.ndarray) -> dict:
    base_norm = {"popularity": "global", "association": "user", "repeat": "user"}
    def baseline(assoc: np.ndarray, repeat_weight: float) -> np.ndarray:
        return fuse_scores(
            {"popularity": pop, "association": assoc, "repeat": repeat},
            {"popularity": 1.0, "association": 0.25, "repeat": repeat_weight},
            base_norm,
        )
    first = baseline(association_legacy, 3.0)
    return {
        "popularity": lambda: pop,
        "first_legacy_repeat3": lambda: first,
        "lift_only_active_repeat3": lambda: baseline(association_active, 3.0),
        "repeat_only_legacy_repeat1": lambda: baseline(association_legacy, 1.0),
        "second_active_repeat1": lambda: baseline(association_active, 1.0),
        "raw_channel": lambda: raw_scores["channel"],
        "raw_channel_hour": lambda: raw_scores["channel_hour"],
        "raw_channel_weekhour": lambda: raw_scores["channel_weekhour"],
        "first_plus_0p5_channel": lambda: first + 0.5 * raw_scores["channel"],
        "first_plus_1p0_channel": lambda: first + 1.0 * raw_scores["channel"],
        "first_plus_0p5_channel_hour": lambda: first + 0.5 * raw_scores["channel_hour"],
        "first_plus_0p5_channel_weekhour": lambda: first + 0.5 * raw_scores["channel_weekhour"],
        "first_plus_0p5_labelled_channel": lambda: first + 0.5 * labelled_channel,
    }


def evaluate_window(window: dict, cfg: dict, p: dict, raw: dict,
                    labelled: dict, channel_ids: np.ndarray,
                    variants_to_score: set[str]) -> tuple[list[dict], dict]:
    t = time.perf_counter()
    fit_start = pd.Timestamp(window["fit_start"])
    cutoff = pd.Timestamp(window["cutoff"])
    end = pd.Timestamp(window["end"])
    schedule = _schedule_for(window, p)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(p["user_ids"], users)
    train = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"],
                             _empty_metadata(), cutoff)
    ev, gt = _all_proxies(cfg, p, schedule, cutoff, end)
    cand_idx = ev["cand_idx"]
    pop = PopularityModel().fit(train, cfg_with(cfg, {"popularity": {"weight": "users"}}),
                                cfg["seed"]).score(uidx, cand_idx, train)
    repeat = RepeatModel().fit(train, cfg_with(cfg, {"repeat": {"half_life_days": 14.0}}),
                               cfg["seed"]).score(uidx, cand_idx, train)
    assoc_cfg = cfg_with(cfg, {"association": {"min_support": 10, "top_k": 100,
                                                "half_life_days": 30.0}})
    associations = {}
    for name, population in (("legacy", "all_indexed_legacy"), ("active", "training_active")):
        c = cfg_with(assoc_cfg, {"association": {"lift_population": population}})
        associations[name] = AssociationModel().fit(train, c, cfg["seed"]).score(uidx, cand_idx, train)
    raw_profile = fit_context_profile(**raw, user_ids=p["user_ids"],
                                      channel_ids=channel_ids, fit_start=fit_start,
                                      cutoff=cutoff, alpha=20.0)
    labelled_profile = fit_context_profile(**labelled, user_ids=p["user_ids"],
                                           channel_ids=channel_ids, fit_start=fit_start,
                                           cutoff=cutoff, alpha=20.0)
    opp = build_opportunity(schedule, ev["candidate_ids"], channel_ids, cutoff, end)
    raw_scores = score_exposure(raw_profile, opp, uidx)
    labelled_channel = score_exposure(labelled_profile, opp, uidx)["channel"]
    variants = _make_variants(pop, associations["legacy"], associations["active"],
                              repeat, raw_scores, labelled_channel)
    expected = set(json.loads(PLAN_PATH.read_text(encoding="utf-8"))["variants"])
    assert set(variants) == expected
    cold_mask = raw_profile.n_user_views[uidx] == 0
    rows = []
    recs_for_examples = {}
    for name, make_scores in variants.items():
        if name not in variants_to_score:
            continue
        scores = make_scores()
        recs, stats = recommend_from_scores(scores, users, ev["candidate_ids"], train,
                                           "allow", pop[0], cold_mask=cold_mask)
        for proxy, target in gt.items():
            m = evaluate_recommendations(recs, target, users, k=5,
                                         n_candidates=ev["n_candidates"])
            rows.append({"window": window["name"], "proxy": proxy, "variant": name,
                         "n_candidates": ev["n_candidates"], "raw_cold_users": int(cold_mask.sum()),
                         "fallback_used": stats["fallback_used"], **m})
        if window["name"] == "early_35d":
            recs_for_examples[name] = recs
        del scores
    profile_stats = {
        "window": window["name"], "train_start": str(fit_start),
        "cutoff": str(cutoff), "eval_end": str(end),
        "raw_train_views": raw_profile.n_training_views,
        "labelled_context_views": labelled_profile.n_training_views,
        "template_raw_cold_users": int(cold_mask.sum()),
        "template_raw_only_users": int(np.count_nonzero(
            (raw_profile.n_user_views[uidx] > 0) & (labelled_profile.n_user_views[uidx] == 0))),
        "n_candidates": ev["n_candidates"],
        "n_candidate_airings": int(opp.n_airings.sum()),
        "n_candidates_with_zero_airings": int(np.count_nonzero(opp.n_airings == 0)),
        "gt_stats": {name: {"eligible": sum(bool(target.get(int(u))) for u in users),
                            "mean_ids_all_users": sum(len(target.get(int(u), set())) for u in users)/len(users)}
                     for name, target in gt.items()},
        "runtime_seconds": round(time.perf_counter() - t, 3),
    }
    return rows, {"stats": profile_stats, "examples": recs_for_examples,
                  "eval": ev, "train": train, "users": users}


def _write_example_rows(output: Path, res: dict, p: dict) -> None:
    selected = json.loads(resolve("midterm/results/selected_model.json").read_text(encoding="utf-8"))
    chosen = [int(selected["users"]["heavy"]), int(selected["users"]["light"])]
    just = {int(x["user_id"]): x for x in selected["users_justification"]}
    ev = res["eval"]
    s0 = p["schedules"]["export_arh_11-20-final.csv"]
    s1 = p["schedules"]["export_arh_21-30-final.csv"]
    ids = {item for recs in res["examples"].values() for user in chosen
           for item in recs[user]}
    lookup = {}
    cutoff = pd.Timestamp("2020-05-18")
    for sched, source in ((s0.loc[s0["start"] < cutoff], "training_schedule"),
                          (s1, "evaluation_schedule_display_only")):
        rows = sched.loc[sched["tv_show_id"].isin(ids),
                         ["tv_show_id", "tv_show_title", "tv_show_category"]]
        for item, g in rows.groupby("tv_show_id", sort=False):
            item = int(item)
            if item in lookup:
                continue
            titles = g["tv_show_title"].astype(str).str.strip()
            cats = g["tv_show_category"].astype(str).str.strip()
            lookup[item] = (next((x for x in titles if x), "(not provided)"),
                            next((x for x in cats if x), "(not provided)"), source)
    out = []
    for user in chosen:
        gt = ev["gt"].get(user, set())
        for name, recs in res["examples"].items():
            for rank, item in enumerate(recs[user], 1):
                title, category, source = lookup.get(item, ("(not provided)", "(not provided)", "not_provided"))
                out.append({"user_id": user, "role": just[user]["role"],
                            "train_labelled_items": just[user]["train_items"],
                            "held_out_items": len(gt), "variant": name, "rank": rank,
                            "tv_show_id": item, "title": title, "category": category,
                            "display_source": source, "observed_in_held_out": item in gt})
    pd.DataFrame(out).to_csv(output, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["selection", "diagnostic"], required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    cfg = load_config()
    p = load_prepared(cfg)
    raw_npz = np.load(RAW_CACHE)
    assert np.array_equal(raw_npz["user_ids"], p["user_ids"])
    raw = {k: raw_npz[k] for k in ("uidx", "channel", "start_ts", "stop_ts", "hour", "weekday")}
    labelled = labelled_view_arrays(p["joined"], p["user_ids"])
    all_channels = [np.unique(raw["channel"])] + [p["schedules"][name]["channel_id"].unique()
                      for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")]
    channel_ids = np.unique(np.concatenate(all_channels)).astype(np.int32)
    if args.phase == "selection":
        windows = plan["selection_windows"]
        variants_to_score = set(plan["variants"])
    else:
        decision_path = ROOT / "selection_decision.json"
        if not decision_path.exists():
            raise RuntimeError("write selection_decision.json before diagnostic evaluation")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("selection_result_sha256") != sha256_file(ROOT / "context_selection.csv"):
            raise RuntimeError("selection results changed since decision")
        windows = [plan["final_diagnostic_window"]]
        variants_to_score = {"popularity", "first_legacy_repeat3", decision["diagnostic_probe"]}
    rows, stats = [], []
    for window in windows:
        new_rows, result = evaluate_window(window, cfg, p, raw, labelled, channel_ids, variants_to_score)
        rows.extend(new_rows)
        stats.append(result["stats"])
        if args.phase == "selection" and window["name"] == "early_35d":
            _write_example_rows(ROOT / "context_examples.csv", result, p)
        all60 = [x for x in new_rows if x["proxy"] == "all_60"]
        print(window["name"], {x["variant"]: round(x["map5_all"], 5) for x in all60}, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    prefix = "context_selection" if args.phase == "selection" else "context_diagnostic"
    pd.DataFrame(rows).to_csv(ROOT / f"{prefix}.csv", index=False)
    (ROOT / f"{prefix}_stats.json").write_text(json.dumps({
        "phase": args.phase, "windows": stats,
        "plan_sha256": sha256_file(PLAN_PATH),
        "raw_cache_sha256": sha256_file(RAW_CACHE),
        "n_channels_union": len(channel_ids),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                        "seed": cfg["seed"]},
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {prefix}, rows={len(rows)}, elapsed={time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

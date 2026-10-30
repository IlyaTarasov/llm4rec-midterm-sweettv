"""SWEET.TV midterm pipeline.

Run from the repository root:
    python -m midterm.src.pipeline --stage prepare
    python -m midterm.src.pipeline --stage validate
    python -m midterm.src.pipeline --stage audit
    python -m midterm.src.pipeline --stage final
    python -m midterm.src.pipeline --stage verify

Stages are intentionally separate so that validation tuning, the later audit
window, and the final test fit are auditable and never silently mixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

# Keep BLAS single-threaded for reproducible ALS arithmetic where possible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import build_item_metadata, candidate_items, window_bounds
from .evaluate import (
    build_candidate_data,
    build_eval_data,
    build_train_data,
    evaluate_scores,
    fit_and_score,
    fuse_scores,
    load_prepared,
    prepare,
    recommend_from_scores,
    rrf_from_scores,
    seen_map,
    summarize_metrics,
)
from .metrics import evaluate_recommendations
from .models import (
    AssociationModel,
    ContentModel,
    MFModel,
    PageRankModel,
    PopularityModel,
    RepeatModel,
)

MODEL_CLASSES = {
    "popularity": PopularityModel,
    "content": ContentModel,
    "association": AssociationModel,
    "mf": MFModel,
    "pagerank": PageRankModel,
    "repeat": RepeatModel,
}
METHOD_ORDER = ["popularity", "content", "association", "mf", "pagerank"]
AUX_ORDER = ["repeat"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_with(base: Dict, updates: Mapping) -> Dict:
    c = copy.deepcopy(base)
    for section, values in updates.items():
        c.setdefault(section, {})
        c[section].update(values)
    return c


def cfg_for_method(base: Dict, name: str, params: Mapping) -> Dict:
    """Apply a method's params whether stored flat or already section-wrapped."""
    updates = params if name in params else {name: params}
    return cfg_with(base, updates)


def fallback_popularity(data, eval_data, cfg: Dict) -> Tuple[PopularityModel, np.ndarray]:
    c = cfg_with(cfg, {"popularity": {"weight": "users"}})
    model = PopularityModel().fit(data, c, int(cfg["seed"]))
    return model, model.pop[eval_data["cand_idx"]]


def tune_method(
    name: str,
    configs: Sequence[Dict],
    data,
    eval_data: Dict,
    users: np.ndarray,
    cfg: Dict,
    fallback_scores: np.ndarray,
    policy: str = "allow",
) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    best: Dict | None = None
    for params in configs:
        c = cfg_with(cfg, params)
        model = MODEL_CLASSES[name]()
        scores, elapsed = fit_and_score(model, data, eval_data, users, c, int(cfg["seed"]))
        res = evaluate_scores(scores, users, eval_data, data, policy, fallback_scores)
        m = res["metrics"]
        row = {
            "method": name,
            "params": json.dumps(params, sort_keys=True, ensure_ascii=False),
            "map5_eligible": m["map5_eligible"],
            "map5_all": m["map5_all"],
            "recall5_eligible": m["recall5_eligible"],
            "hit_rate_eligible": m["hit_rate_eligible"],
            "n_eligible": m["n_eligible"],
            "catalog_coverage": m["catalog_coverage"],
            "runtime_seconds": round(elapsed, 3),
        }
        rows.append(row)
        key = (m["map5_eligible"], m["map5_all"])
        if best is None or key > best["key"] or (key == best["key"] and elapsed < best["runtime"]):
            best = {
                "params": params,
                "cfg": c,
                "model": model,
                "scores": scores,
                "metrics": m,
                "runtime": elapsed,
                "key": key,
            }
        print(f"  {name:12s} {json.dumps(params, ensure_ascii=False):55s} "
              f"MAP5={m['map5_eligible']:.5f} all={m['map5_all']:.5f} ({elapsed:.1f}s)", flush=True)
    assert best is not None
    return rows, best


def tune_validation(cfg: Dict, data, eval_data: Dict, users: np.ndarray) -> Dict:
    _, fallback_scores = fallback_popularity(data, eval_data, cfg)
    grids = {
        "popularity": [
            {"popularity": {"weight": "users"}},
            {"popularity": {"weight": "views"}},
        ],
        "content": [
            {"content": {"half_life_days": 14, "use_cast": True}},
            {"content": {"half_life_days": 30, "use_cast": True}},
            {"content": {"half_life_days": 60, "use_cast": True}},
            {"content": {"half_life_days": 30, "use_cast": False}},
        ],
        "association": [
            {"association": {"min_support": 3, "top_k": 100}},
            {"association": {"min_support": 5, "top_k": 100}},
            {"association": {"min_support": 10, "top_k": 100}},
            {"association": {"min_support": 5, "top_k": 200}},
        ],
        "mf": [
            {"mf": {"factors": 32, "alpha": 20.0, "weight": "count"}},
            {"mf": {"factors": 64, "alpha": 20.0, "weight": "count"}},
            {"mf": {"factors": 64, "alpha": 10.0, "weight": "count"}},
            {"mf": {"factors": 64, "alpha": 20.0, "weight": "binary"}},
        ],
        "pagerank": [
            {"pagerank": {"alpha": 0.70, "top_k": 50}},
            {"pagerank": {"alpha": 0.85, "top_k": 50}},
            {"pagerank": {"alpha": 0.95, "top_k": 50}},
            {"pagerank": {"alpha": 0.85, "top_k": 30}},
        ],
    }
    all_rows: List[Dict] = []
    best: Dict[str, Dict] = {}
    for name in METHOD_ORDER:
        print(f"Tuning {name} ...", flush=True)
        rows, b = tune_method(name, grids[name], data, eval_data, users, cfg, fallback_scores, "allow")
        all_rows.extend(rows)
        best[name] = b
    return {"rows": all_rows, "best": best, "fallback_scores": fallback_scores}


def tune_repeat_and_fusion(
    cfg: Dict,
    best: Mapping[str, Dict],
    data,
    eval_data: Dict,
    users: np.ndarray,
    fallback_scores: np.ndarray,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Tune the auxiliary repeat prior and the popularity/association/repeat fusion."""
    repeat_rows: List[Dict] = []
    repeat_scores: Dict[float, np.ndarray] = {}
    repeat_metrics: Dict[float, Dict] = {}
    for hl in (7.0, 14.0, 30.0, 60.0):
        c = cfg_with(cfg, {"repeat": {"half_life_days": hl}})
        model = RepeatModel()
        scores, elapsed = fit_and_score(model, data, eval_data, users, c, int(cfg["seed"]))
        repeat_scores[hl] = scores
        res = evaluate_scores(scores, users, eval_data, data, "allow", fallback_scores)
        m = res["metrics"]
        repeat_metrics[hl] = m
        repeat_rows.append({
            "method": "repeat",
            "half_life_days": hl,
            "map5_eligible": m["map5_eligible"],
            "map5_all": m["map5_all"],
            "recall5_eligible": m["recall5_eligible"],
            "hit_rate_eligible": m["hit_rate_eligible"],
            "runtime_seconds": round(elapsed, 3),
        })
        print(f"  repeat       half_life={hl:<5} MAP5={m['map5_eligible']:.5f} all={m['map5_all']:.5f}", flush=True)

    fusion_rows: List[Dict] = []
    best_fusion: Dict | None = None
    pop_scores = best["popularity"]["scores"]
    assoc_scores = best["association"]["scores"]
    normalizations = {"popularity": "global", "association": "user", "repeat": "user"}
    for hl, rep_scores in repeat_scores.items():
        for wa in (0.0, 0.1, 0.25, 0.5, 1.0):
            for wr in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
                sm = {"popularity": pop_scores, "association": assoc_scores, "repeat": rep_scores}
                weights = {"popularity": 1.0, "association": wa, "repeat": wr}
                fused = fuse_scores(sm, weights, normalizations)
                res = evaluate_scores(fused, users, eval_data, data, "allow", fallback_scores)
                m = res["metrics"]
                row = {
                    "strategy": "fusion",
                    "half_life_days": hl,
                    "w_popularity": 1.0,
                    "w_association": wa,
                    "w_repeat": wr,
                    "map5_eligible": m["map5_eligible"],
                    "map5_all": m["map5_all"],
                    "recall5_eligible": m["recall5_eligible"],
                    "hit_rate_eligible": m["hit_rate_eligible"],
                }
                fusion_rows.append(row)
                key = (m["map5_eligible"], m["map5_all"])
                if best_fusion is None or key > best_fusion["key"]:
                    best_fusion = {
                        "scores": fused, "metrics": m, "recs": res["recs"],
                        "weights": weights, "normalizations": normalizations,
                        "half_life_days": hl, "key": key,
                    }
    assert best_fusion is not None
    print(f"  fusion best: half_life={best_fusion['half_life_days']} "
          f"w_assoc={best_fusion['weights']['association']} w_repeat={best_fusion['weights']['repeat']} "
          f"MAP5={best_fusion['metrics']['map5_eligible']:.5f}", flush=True)
    best_fusion["repeat_scores"] = repeat_scores
    best_fusion["repeat_metrics"] = repeat_metrics
    return repeat_rows, fusion_rows, best_fusion


def repeat_policy_table(best: Mapping[str, Dict], data, eval_data: Dict, users: np.ndarray,
                        fallback_scores: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    chosen: Dict = {}
    for name in METHOD_ORDER:
        for policy in ("allow", "exclude"):
            res = evaluate_scores(best[name]["scores"], users, eval_data, data, policy, fallback_scores)
            m = res["metrics"]
            rows.append({"method": name, "repeat_policy": policy, **m})
            chosen[(name, policy)] = res
    return pd.DataFrame(rows), chosen


def hybrid_weights_list() -> List[Dict[str, float]]:
    return [
        {"popularity": 1.0},
        {"content": 1.0},
        {"association": 1.0},
        {"mf": 1.0},
        {"pagerank": 1.0},
        {"popularity": 0.5, "content": 0.5},
        {"popularity": 0.5, "mf": 1.0},
        {"popularity": 0.5, "pagerank": 0.5},
        {"content": 0.5, "mf": 1.0},
        {"association": 0.5, "mf": 1.0},
        {"mf": 1.0, "pagerank": 0.5},
        {"popularity": 0.5, "content": 0.5, "mf": 1.0},
        {"popularity": 0.5, "mf": 1.0, "pagerank": 0.5},
        {"popularity": 0.33, "content": 0.33, "association": 0.33, "mf": 0.66, "pagerank": 0.33},
        {"popularity": 0.5, "content": 0.5, "association": 0.5, "mf": 1.0, "pagerank": 0.5},
    ]


def evaluate_hybrids(score_map: Mapping[str, np.ndarray], data, eval_data: Dict,
                     users: np.ndarray, fallback_scores: np.ndarray, rrf_k: int = 60) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    best = None
    for policy in ("allow", "exclude"):
        for weights in hybrid_weights_list():
            scores = rrf_from_scores(score_map, weights, rrf_k)
            res = evaluate_scores(scores, users, eval_data, data, policy, fallback_scores)
            m = res["metrics"]
            row = {
                "strategy": "hybrid",
                "weights": json.dumps(weights, sort_keys=True),
                "repeat_policy": policy,
                **m,
            }
            rows.append(row)
            key = (m["map5_eligible"], m["map5_all"])
            if best is None or key > best["key"]:
                best = {"weights": weights, "policy": policy, "scores": scores,
                        "metrics": m, "recs": res["recs"], "key": key}
    return pd.DataFrame(rows), best


def choose_two_users(submission: pd.DataFrame, data, eval_data: Dict) -> Tuple[Dict, List[Dict]]:
    users = submission["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(data.user_ids, users)
    hist = np.diff(data.X_binary.indptr)[uidx]
    gt_counts = np.array([len(eval_data["gt"].get(int(u), set())) for u in users])
    heavy_pos = int(np.argmax(hist))
    heavy = int(users[heavy_pos])
    cold_positions = np.flatnonzero(hist == 0)
    if len(cold_positions):
        # Prefer a cold user with held-out activity so the example is informative.
        with_gt = [p for p in cold_positions if gt_counts[p] > 0]
        cold_pos = int(with_gt[0]) if with_gt else int(cold_positions[0])
        role = "cold (no training interactions)"
    else:
        # Fall back to the lightest warm user with held-out activity.
        eligible = np.flatnonzero(gt_counts > 0)
        pool = eligible if len(eligible) else np.arange(len(users))
        cold_pos = int(pool[np.argmin(hist[pool])])
        role = "light warm (no cold user available)"
    light = int(users[cold_pos])
    info = [
        {"user_id": heavy, "role": "heavy warm",
         "train_items": int(hist[heavy_pos]), "held_out_items": int(gt_counts[heavy_pos])},
        {"user_id": light, "role": role,
         "train_items": int(hist[cold_pos]), "held_out_items": int(gt_counts[cold_pos])},
    ]
    return {"heavy": heavy, "light": light}, info


def examples_table(users_info: List[Dict], recs_by_method: Mapping[str, Mapping[int, Sequence[int]]],
                   eval_data: Dict, item_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for info in users_info:
        u = int(info["user_id"])
        gt = eval_data["gt"].get(u, set())
        for method, recs in recs_by_method.items():
            for rank, item in enumerate(recs.get(u, []), start=1):
                meta = item_meta.loc[item] if item in item_meta.index else None
                rows.append({
                    "user_id": u,
                    "user_role": info["role"],
                    "train_items": info["train_items"],
                    "held_out_items": info["held_out_items"],
                    "method": method,
                    "rank": rank,
                    "tv_show_id": int(item),
                    "title": "" if meta is None else str(meta["title"]),
                    "category": "" if meta is None else str(meta["category"]),
                    "in_held_out": bool(int(item) in gt),
                })
    return pd.DataFrame(rows)


def stage_validate(cfg: Dict) -> None:
    t0 = time.time()
    p = load_prepared(cfg)
    b = window_bounds(cfg)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    item_meta = build_item_metadata([p["schedules"]["export_arh_11-20-final.csv"]], b["val_start"])
    data = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], item_meta, b["val_start"])
    eval_data = build_eval_data(cfg, p["joined"], p["schedules"]["export_arh_21-30-final.csv"],
                                p["user_ids"], p["item_ids"], item_meta, b["val_start"], b["val_end"])
    print(f"Validation: train pairs={len(data.interactions)} users={data.n_users} items={data.n_items}; "
          f"candidates={eval_data['n_candidates']} gt_users={eval_data['n_users_with_gt']} "
          f"candidate_meta={eval_data['metadata_coverage']}", flush=True)

    tuned = tune_validation(cfg, data, eval_data, users)
    best = tuned["best"]
    fallback_scores = tuned["fallback_scores"]
    pd.DataFrame(tuned["rows"]).to_csv(resolve(cfg["results_dir"]) / "tuning_validation.csv", index=False)

    repeat_rows, fusion_rows, fusion_best = tune_repeat_and_fusion(
        cfg, best, data, eval_data, users, fallback_scores
    )
    pd.DataFrame(repeat_rows).to_csv(resolve(cfg["results_dir"]) / "repeat_tuning_validation.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(resolve(cfg["results_dir"]) / "fusion_grid_validation.csv", index=False)

    policy_df, policy_recs = repeat_policy_table(best, data, eval_data, users, fallback_scores)
    best_hl = fusion_best["half_life_days"]
    repeat_policy_rows = []
    for policy in ("allow", "exclude"):
        res = evaluate_scores(fusion_best["repeat_scores"][best_hl], users, eval_data, data, policy, fallback_scores)
        policy_recs[("repeat", policy)] = res
        repeat_policy_rows.append({"method": "repeat", "repeat_policy": policy, **res["metrics"]})
    policy_df = pd.concat([policy_df, pd.DataFrame(repeat_policy_rows)], ignore_index=True)
    policy_df.to_csv(resolve(cfg["results_dir"]) / "repeat_policy_comparison.csv", index=False)

    score_map = {name: best[name]["scores"] for name in METHOD_ORDER}
    hybrid_df, hybrid_best = evaluate_hybrids(score_map, data, eval_data, users, fallback_scores,
                                              rrf_k=int(cfg["hybrid"].get("rrf_k", 60)))
    hybrid_df.to_csv(resolve(cfg["results_dir"]) / "hybrid_grid_validation.csv", index=False)

    individual_candidates = []
    for name in METHOD_ORDER + AUX_ORDER:
        for policy in ("allow", "exclude"):
            res = policy_recs[(name, policy)]
            m = res["metrics"]
            runtime = float(best[name]["runtime"]) if name in best else 0.0
            individual_candidates.append({
                "strategy": name, "repeat_policy": policy, "scores": None,
                "metrics": m, "recs": res["recs"],
                "key": (m["map5_eligible"], m["map5_all"], -runtime),
                "runtime": runtime,
            })
    rrf_candidate = {
        "strategy": "hybrid", "repeat_policy": hybrid_best["policy"], "scores": hybrid_best["scores"],
        "metrics": hybrid_best["metrics"], "recs": hybrid_best["recs"],
        "key": (hybrid_best["metrics"]["map5_eligible"], hybrid_best["metrics"]["map5_all"], -1e9),
        "runtime": float(sum(best[n]["runtime"] for n in METHOD_ORDER)),
        "weights": hybrid_best["weights"],
    }
    fm = fusion_best["metrics"]
    fusion_candidate = {
        "strategy": "fusion", "repeat_policy": "allow", "scores": fusion_best["scores"],
        "metrics": fm, "recs": fusion_best["recs"],
        "key": (fm["map5_eligible"], fm["map5_all"], -1e9),
        "runtime": float(best["popularity"]["runtime"] + best["association"]["runtime"]),
        "weights": fusion_best["weights"],
        "normalizations": fusion_best["normalizations"],
        "half_life_days": best_hl,
    }
    selected = max(individual_candidates + [rrf_candidate, fusion_candidate], key=lambda x: x["key"])
    chosen_policy = selected["repeat_policy"]

    comparison_results = {}
    for name in METHOD_ORDER + AUX_ORDER:
        comparison_results[name] = policy_recs[(name, chosen_policy)]
    comparison_results["hybrid"] = {
        "metrics": hybrid_best["metrics"], "recs": hybrid_best["recs"],
        "runtime_seconds": float(sum(best[n]["runtime"] for n in METHOD_ORDER)),
    }
    comparison_results["fusion"] = {
        "metrics": fusion_best["metrics"], "recs": fusion_best["recs"],
        "runtime_seconds": fusion_candidate["runtime"],
    }
    comparison_df = summarize_metrics(comparison_results, users, eval_data, data)
    comparison_df.to_csv(resolve(cfg["results_dir"]) / "validation_comparison.csv", index=False)

    users_sel, users_info = choose_two_users(p["submission"], data, eval_data)
    recs_by_method = {name: comparison_results[name]["recs"] for name in METHOD_ORDER + AUX_ORDER}
    recs_by_method["hybrid"] = comparison_results["hybrid"]["recs"]
    recs_by_method["fusion"] = comparison_results["fusion"]["recs"]
    ex_df = examples_table(users_info, recs_by_method, eval_data, item_meta)
    ex_df.to_csv(resolve(cfg["results_dir"]) / "examples_validation.csv", index=False)
    with open(resolve(cfg["results_dir"]) / "examples_validation.json", "w", encoding="utf-8") as fh:
        json.dump({"users": users_info, "rows": ex_df.to_dict(orient="records")}, fh, indent=2, ensure_ascii=False)

    method_configs = {name: best[name]["params"] for name in METHOD_ORDER}
    method_configs["repeat"] = {"half_life_days": best_hl}
    method_metrics = {name: best[name]["metrics"] for name in METHOD_ORDER}
    method_metrics["repeat"] = fusion_best["repeat_metrics"][best_hl]

    if selected["strategy"] == "fusion":
        components = {n: method_configs[n] for n, w in selected["weights"].items() if w > 0}
        fusion_weights = {n: {"weight": float(w), "normalize": selected["normalizations"][n]}
                          for n, w in selected["weights"].items() if w > 0}
        rrf_weights = {}
    elif selected["strategy"] == "hybrid":
        components = {n: method_configs[n] for n in selected["weights"]}
        fusion_weights = {}
        rrf_weights = selected["weights"]
    else:
        components = {selected["strategy"]: method_configs[selected["strategy"]]}
        fusion_weights = {}
        rrf_weights = {}

    selected_doc = {
        "created_utc": utc_now(),
        "strategy": selected["strategy"],
        "repeat_policy": chosen_policy,
        "seed": int(cfg["seed"]),
        "join": cfg["join"],
        "fallback": {"method": "popularity", "weight": "users"},
        "method_configs": method_configs,
        "method_validation_metrics": method_metrics,
        "components": components,
        "weights": rrf_weights,
        "fusion_weights": fusion_weights,
        "rrf_k": int(cfg["hybrid"].get("rrf_k", 60)),
        "validation_metrics": selected["metrics"],
        "validation_runtime_seconds": selected["runtime"],
        "selection_rule": "max map5_eligible, then map5_all, then lower runtime; tuning only on weeks 21-25",
        "validation_window": [str(b["val_start"]), str(b["val_end"])],
        "train_window": [str(b["fit_start"]), str(b["val_start"])],
        "metadata_cutoff": str(b["val_start"]),
        "metadata_sources": ["export_arh_11-20-final.csv"],
        "candidate_metadata_coverage": eval_data["metadata_coverage"],
        "users": users_sel,
        "users_justification": users_info,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "seed": int(cfg["seed"]),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(resolve(cfg["results_dir"]) / "selected_model.json", "w", encoding="utf-8") as fh:
        json.dump(selected_doc, fh, indent=2, ensure_ascii=False)
    print(f"Selected: {selected['strategy']} policy={chosen_policy} "
          f"MAP5={selected['metrics']['map5_eligible']:.5f} "
          f"(all={selected['metrics']['map5_all']:.5f})", flush=True)
    print(f"Validation stage finished in {time.time() - t0:.1f}s", flush=True)


def load_selected(cfg: Dict) -> Dict:
    with open(resolve(cfg["results_dir"]) / "selected_model.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def fit_components(selected: Mapping, data, cfg: Dict) -> Dict[str, object]:
    models = {}
    for name, params in selected["components"].items():
        c = cfg_for_method(cfg, name, params)
        model = MODEL_CLASSES[name]()
        model.fit(data, c, int(cfg["seed"]))
        models[name] = model
    return models


def score_components(models: Mapping[str, object], selected: Mapping, data,
                     uidx: np.ndarray, cand_idx: np.ndarray) -> np.ndarray:
    if selected["strategy"] == "fusion":
        score_map = {name: model.score(uidx, cand_idx, data) for name, model in models.items()}
        weights = {name: float(v["weight"]) for name, v in selected["fusion_weights"].items()}
        norms = {name: str(v["normalize"]) for name, v in selected["fusion_weights"].items()}
        return fuse_scores(score_map, weights, norms)
    if selected["strategy"] == "hybrid":
        score_map = {name: model.score(uidx, cand_idx, data) for name, model in models.items()}
        return rrf_from_scores(score_map, selected["weights"], int(selected.get("rrf_k", 60)))
    return models[selected["strategy"]].score(uidx, cand_idx, data)


def stage_audit(cfg: Dict) -> None:
    t0 = time.time()
    p = load_prepared(cfg)
    selected = load_selected(cfg)
    b = window_bounds(cfg)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    item_meta = build_item_metadata(
        [p["schedules"]["export_arh_11-20-final.csv"], p["schedules"]["export_arh_21-30-final.csv"]],
        b["val_end"],
    )
    data = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], item_meta, b["val_end"])
    eval_data = build_eval_data(cfg, p["joined"], p["schedules"]["export_arh_21-30-final.csv"],
                                p["user_ids"], p["item_ids"], item_meta, b["val_end"], b["audit_end"])
    _, fallback_scores = fallback_popularity(data, eval_data, cfg)
    print(f"Audit: train pairs={len(data.interactions)}; candidates={eval_data['n_candidates']} "
          f"gt_users={eval_data['n_users_with_gt']}", flush=True)

    results = {}
    for name in METHOD_ORDER + AUX_ORDER:
        c = cfg_for_method(cfg, name, selected["method_configs"][name])
        model = MODEL_CLASSES[name]()
        scores, elapsed = fit_and_score(model, data, eval_data, users, c, int(cfg["seed"]))
        res = evaluate_scores(scores, users, eval_data, data, selected["repeat_policy"], fallback_scores)
        res["runtime_seconds"] = round(elapsed, 3)
        results[name] = res
    models = fit_components(selected, data, cfg)
    uidx = np.searchsorted(data.user_ids, users)
    t1 = time.time()
    sel_scores = score_components(models, selected, data, uidx, eval_data["cand_idx"])
    res = evaluate_scores(sel_scores, users, eval_data, data, selected["repeat_policy"], fallback_scores)
    res["runtime_seconds"] = round(time.time() - t1, 3)
    results["selected"] = res
    if selected["strategy"] == "fusion":
        results["fusion"] = res
    comparison = summarize_metrics(results, users, eval_data, data)
    comparison.to_csv(resolve(cfg["results_dir"]) / "audit_comparison.csv", index=False)

    users_sel, users_info = choose_two_users(p["submission"], data, eval_data)
    recs_by_method = {name: results[name]["recs"] for name in METHOD_ORDER + AUX_ORDER}
    recs_by_method["selected"] = results["selected"]["recs"]
    if "fusion" in results:
        recs_by_method["fusion"] = results["fusion"]["recs"]
    ex_df = examples_table(users_info, recs_by_method, eval_data, item_meta)
    ex_df.to_csv(resolve(cfg["results_dir"]) / "examples_audit.csv", index=False)
    with open(resolve(cfg["results_dir"]) / "audit_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "created_utc": utc_now(),
            "audit_window": [str(b["val_end"]), str(b["audit_end"])],
            "train_window": [str(b["fit_start"]), str(b["val_end"])],
            "metadata_cutoff": str(b["val_end"]),
            "candidate_metadata_coverage": eval_data["metadata_coverage"],
            "selected_strategy": selected["strategy"],
            "selected_metrics": results["selected"]["metrics"],
            "users": users_info,
            "elapsed_seconds": round(time.time() - t0, 2),
        }, fh, indent=2, ensure_ascii=False)
    print(f"Audit selected MAP5={results['selected']['metrics']['map5_eligible']:.5f} "
          f"(all={results['selected']['metrics']['map5_all']:.5f})", flush=True)


def stage_final(cfg: Dict) -> None:
    t0 = time.time()
    p = load_prepared(cfg)
    selected = load_selected(cfg)
    b = window_bounds(cfg)
    submission = p["submission"]
    users = submission["user_id"].to_numpy(dtype=np.uint64)
    item_meta = build_item_metadata(
        [p["schedules"]["export_arh_11-20-final.csv"], p["schedules"]["export_arh_21-30-final.csv"]],
        b["audit_end"],
    )
    data = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], item_meta, b["audit_end"])
    eval_data = build_candidate_data(
        p["schedules"]["export_arh_31-42-final.csv"], p["item_ids"], item_meta,
        b["audit_end"], b["test_end"],
    )
    _, fallback_scores = fallback_popularity(data, eval_data, cfg)
    print(f"Final: train pairs={len(data.interactions)} users={data.n_users}; "
          f"test candidates={eval_data['n_candidates']}", flush=True)

    models = fit_components(selected, data, cfg)
    uidx = np.searchsorted(data.user_ids, users)
    scores = score_components(models, selected, data, uidx, eval_data["cand_idx"])
    recs, stats = recommend_from_scores(scores, users, eval_data["candidate_ids"], data,
                                        selected["repeat_policy"], fallback_scores, k=5)
    rows = []
    for u in users:
        ids = recs[int(u)]
        if len(ids) != 5:
            raise RuntimeError(f"user {u} has {len(ids)} recommendations")
        rows.append({"user_id": int(u), "tv_show_id": " ".join(str(int(x)) for x in ids)})
    out = pd.DataFrame(rows, columns=["user_id", "tv_show_id"])
    out_path = resolve("midterm/submission.csv")
    out.to_csv(out_path, index=False, lineterminator="\n")
    digest = sha256_file(out_path)
    selected_path = resolve(cfg["results_dir"]) / "selected_model.json"
    manifest = {
        "created_utc": utc_now(),
        "strategy": selected["strategy"],
        "components": selected["components"],
        "weights": selected["weights"],
        "fusion_weights": selected.get("fusion_weights", {}),
        "repeat_policy": selected["repeat_policy"],
        "seed": int(cfg["seed"]),
        "train_window": [str(b["fit_start"]), str(b["audit_end"])],
        "test_window": [str(b["audit_end"]), str(b["test_end"])],
        "metadata_cutoff": str(b["audit_end"]),
        "candidate_metadata_coverage": eval_data["metadata_coverage"],
        "n_users": int(len(out)),
        "n_test_candidates": int(eval_data["n_candidates"]),
        "submission_path": str(out_path),
        "submission_sha256": digest,
        "selected_model_path": str(selected_path),
        "selected_model_sha256": sha256_file(selected_path),
        "fallback": {"method": "popularity", "weight": "users"},
        "recommendation_stats": stats,
        "runtime_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(resolve(cfg["results_dir"]) / "final_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    run_dir = resolve("midterm/runs") / datetime.now().strftime("%Y-%m-%d-%H%M-final")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    with open(run_dir / "selected_model.json", "w", encoding="utf-8") as fh:
        json.dump(selected, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path} sha256={digest} in {time.time() - t0:.1f}s", flush=True)


def stage_verify(cfg: Dict) -> None:
    from .verify import verify_submission
    report = verify_submission(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prepare", "validate", "audit", "final", "verify"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    resolve(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    resolve(cfg["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        print(json.dumps(prepare(cfg), indent=2, ensure_ascii=False))
    elif args.stage == "validate":
        stage_validate(cfg)
    elif args.stage == "audit":
        stage_audit(cfg)
    elif args.stage == "final":
        stage_final(cfg)
    elif args.stage == "verify":
        stage_verify(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
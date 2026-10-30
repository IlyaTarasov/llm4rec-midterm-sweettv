"""Structural and semantic verification of the final Kaggle submission CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import resolve
from .data import build_item_metadata, candidate_items, window_bounds
from .evaluate import build_train_data, load_prepared
from .models import PopularityModel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_submission(cfg: Dict) -> Dict:
    data_dir = resolve(cfg["data_dir"])
    template = pd.read_csv(data_dir / "submission.csv", dtype={"user_id": "uint64", "tv_show_id": "string"})
    path = resolve("midterm/submission.csv")
    checks: List[Dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("submission_exists", path.exists(), str(path))
    if not path.exists():
        report = {"ok": False, "checks": checks}
        with open(resolve(cfg["results_dir"]) / "submission_verification.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        return report

    sub = pd.read_csv(path, dtype={"user_id": "uint64", "tv_show_id": "string"}, keep_default_na=False)
    check("columns", list(sub.columns) == ["user_id", "tv_show_id"], str(list(sub.columns)))
    check("row_count", len(sub) == len(template), f"{len(sub)} vs {len(template)}")
    check("user_order", bool(np.array_equal(sub["user_id"].to_numpy(), template["user_id"].to_numpy())),
          "user ids and order match template")
    check("no_missing", (not sub.isna().any().any()) and bool((sub["tv_show_id"].astype(str).str.len() > 0).all()),
          "no NaN/empty cells")

    test_schedule = pd.read_pickle(resolve(cfg["artifacts_dir"]) / "schedule_export_arh_31-42-final.pkl")
    b = window_bounds(cfg)
    allowed = set(int(x) for x in candidate_items(test_schedule, b["audit_end"], b["test_end"]))
    token_rows = sub["tv_show_id"].astype(str).str.split()
    lengths = token_rows.apply(len)
    check("five_tokens", bool((lengths == 5).all()), f"length counts={lengths.value_counts().to_dict()}")
    parsed: List[List[int]] = []
    bad = 0
    for toks in token_rows:
        try:
            vals = [int(t) for t in toks]
        except Exception:
            bad += 1
            parsed.append([])
            continue
        parsed.append(vals)
    check("integer_tokens", bad == 0, f"unparseable rows={bad}")
    check("nonzero_ids", all(all(v > 0 for v in vals) for vals in parsed), "all ids > 0")
    check("unique_ids", all(len(set(vals)) == 5 for vals in parsed), "five unique ids per row")
    check("allowed_test_ids", all(all(v in allowed for v in vals) for vals in parsed),
          f"allowed pool size={len(allowed)}")

    selected_path = resolve(cfg["results_dir"]) / "selected_model.json"
    manifest_path = resolve(cfg["results_dir"]) / "final_manifest.json"
    check("selected_model_exists", selected_path.exists(), str(selected_path))
    check("final_manifest_exists", manifest_path.exists(), str(manifest_path))
    digest = _sha256(path)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        check("submission_hash_matches_manifest", manifest.get("submission_sha256") == digest,
              f"{digest} vs {manifest.get('submission_sha256')}")
        if selected_path.exists():
            check("selected_model_hash_matches_manifest",
                  manifest.get("selected_model_sha256") == _sha256(selected_path),
                  "selected_model.json hash")
    else:
        manifest = {}
    with open(resolve(cfg["results_dir"]) / "selected_model.json", "r", encoding="utf-8") as fh:
        selected = json.load(fh)

    # Semantic checks that need the full training history.
    p = load_prepared(cfg)
    item_meta = build_item_metadata(
        [p["schedules"]["export_arh_11-20-final.csv"], p["schedules"]["export_arh_21-30-final.csv"]],
        b["audit_end"],
    )
    data = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], item_meta, b["audit_end"])
    users = sub["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(data.user_ids, users)
    hist = np.diff(data.X_binary.indptr)[uidx]
    if selected.get("repeat_policy") == "exclude":
        seen = {}
        for u, g in data.interactions.groupby("user_id", sort=False):
            seen[int(u)] = set(int(x) for x in g["item_id"].unique())
        violations = sum(1 for u, vals in zip(users, parsed) if seen.get(int(u), set()) & set(vals))
        check("repeat_policy_exclude", violations == 0, f"users with a repeat={violations}")
    else:
        check("repeat_policy_allow", True, "repeats explicitly allowed")

    # Cold-user fallback must be the same training-only popularity list everywhere.
    fallback_cfg = {**cfg, "popularity": {**cfg.get("popularity", {}), "weight": "users"}}
    pop = PopularityModel().fit(data, fallback_cfg, int(cfg["seed"]))
    cand_ids = np.array(sorted(allowed), dtype=np.int64)
    cand_idx = np.searchsorted(data.item_ids, cand_ids)
    fb_scores = pop.pop[cand_idx]
    order = np.argsort(-fb_scores, kind="stable")[:5]
    expected = [int(cand_ids[i]) for i in order]
    cold_positions = np.flatnonzero(hist == 0)
    mismatches = 0
    for pos in cold_positions:
        if parsed[int(pos)] != expected:
            mismatches += 1
    check("cold_fallback_popularity", mismatches == 0,
          f"cold users={len(cold_positions)}, mismatches={mismatches}")

    # Deterministic tie behaviour: equal scores resolve to the smaller programme id.
    from .metrics import topk_items
    tie_scores = np.array([1.0, 0.0, 1.0, 1.0], dtype=float)
    tie_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    tie_order = topk_items(tie_scores, tie_ids, k=3)
    check("deterministic_ties", tie_order == [10, 30, 40], f"got {tie_order}")

    report = {
        "ok": all(c["ok"] for c in checks),
        "submission_path": str(path),
        "submission_sha256": digest,
        "n_rows": int(len(sub)),
        "n_cold_users": int(len(cold_positions)),
        "checks": checks,
    }
    with open(resolve(cfg["results_dir"]) / "submission_verification.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report
"""Bounded corrected-target optimization after the third SWEET.TV submission."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .context_experiment import _empty_metadata, _schedule_for
from .corrected_target_experiment import ROOT, _qualified_training
from .data import candidate_items
from .evaluate import build_train_data, fuse_scores, load_prepared, recommend_from_scores
from .metrics import ap_at_k, evaluate_recommendations
from .models import AssociationModel, PageRankModel, PopularityModel, RepeatModel
from .pipeline import cfg_for_method, cfg_with, sha256_file
from .qualified_airings import qualify_airings, top5_frequency

PLAN = ROOT / "fourth_pass_plan.json"


def config_id(half_life, repeat_weight, association_weight):
    return f"h{int(half_life)}_r{repeat_weight:g}_a{association_weight:g}"


def opportunity_factor(p, w, candidates):
    start, cutoff, end = map(pd.Timestamp, (w["fit_start"], w["cutoff"], w["end"]))
    past_schedule = pd.concat([p["schedules"]["export_arh_11-20-final.csv"],
                               p["schedules"]["export_arh_21-30-final.csv"]], ignore_index=True)
    if w["schedule"] == "31-42":
        future_schedule = p["schedules"]["export_arh_31-42-final.csv"][
            ["tv_show_id", "channel_id", "start", "duration"]]
    else:
        future_schedule = _schedule_for(w, p)
    def count(schedule, lo, hi):
        x = schedule.loc[(schedule["tv_show_id"] != 0) & (schedule["start"] >= lo)
                         & (schedule["start"] < hi) & (schedule["end"] <= hi),
                         ["tv_show_id", "channel_id", "start", "duration"]].drop_duplicates()
        return x.groupby("tv_show_id").size().reindex(candidates, fill_value=0).to_numpy(dtype=float)
    past = count(past_schedule, start, cutoff)
    # _schedule_for keeps duration/start but not end; compute end here.
    future_schedule = future_schedule.copy()
    future_schedule["end"] = future_schedule["start"] + pd.to_timedelta(future_schedule["duration"], unit="s")
    future = count(future_schedule, cutoff, end)
    past_rate = np.minimum(past / ((cutoff-start).total_seconds()/86400), 2.0)
    future_rate = np.minimum(future / ((end-cutoff).total_seconds()/86400), 2.0)
    return np.clip(np.sqrt((future_rate+0.1)/(past_rate+0.1)), 0.5, 1.5)


def prepare_window(p, cfg, w, need_ppr=False):
    start, cutoff, end = map(pd.Timestamp, (w["fit_start"], w["cutoff"], w["end"]))
    schedule = _schedule_for(w, p)
    candidates = candidate_items(schedule, cutoff, end)
    cand_idx = np.searchsorted(p["item_ids"], candidates)
    users = p["submission"]["user_id"].to_numpy(dtype=np.uint64)
    uidx = np.searchsorted(p["user_ids"], users)
    qtrain = qualify_airings(p["joined"], start, cutoff)
    qeval = qualify_airings(p["joined"], cutoff, end)
    gt, _ = top5_frequency(qeval, candidates)
    eligible = np.array([u for u in users if len(gt.get(int(u), ())) == 5], dtype=np.uint64)
    one = p["joined"].loc[
        (p["joined"]["sched_start"] >= cutoff) & (p["joined"]["sched_start"] < end)
        & (p["joined"]["sched_end"] <= end) & (p["joined"]["view_end"] <= end)
        & (p["joined"]["overlap_fraction"] >= 0.8),
        ["user_id", "item_id", "channel_id", "sched_start", "sched_end"],
    ].drop_duplicates()
    gt_single, _ = top5_frequency(one, candidates)
    single_eligible = np.array([u for u in users if len(gt_single.get(int(u), ())) == 5], dtype=np.uint64)
    old = build_train_data(cfg, p["joined"], p["user_ids"], p["item_ids"], _empty_metadata(), cutoff)
    train = _qualified_training(qtrain, cfg, p, cutoff)
    pop = PopularityModel().fit(train, cfg_with(cfg, {"popularity":{"weight":"users"}}), cfg["seed"]).score(uidx, cand_idx, train)
    fallback = PopularityModel().fit(old, cfg_with(cfg, {"popularity":{"weight":"users"}}), cfg["seed"]).score(uidx, cand_idx, old)[0]
    assoc_cfg = cfg_with(cfg, {"association":{"min_support":10,"top_k":100,"half_life_days":30.0,"lift_population":"training_active"}})
    assoc = AssociationModel().fit(train, assoc_cfg, cfg["seed"]).score(uidx, cand_idx, train)
    repeat = {h: RepeatModel().fit(train, cfg_with(cfg, {"repeat":{"half_life_days":h}}), cfg["seed"]).score(uidx, cand_idx, train)
              for h in (7.0,14.0,28.0)}
    cold = np.diff(train.X_binary.indptr)[uidx] == 0
    ppr = None
    if need_ppr:
        selected = json.loads(resolve("midterm/results/selected_model.json").read_text(encoding="utf-8"))
        pcfg = cfg_for_method(cfg, "pagerank", selected["method_configs"]["pagerank"])
        ppr = PageRankModel().fit(train, pcfg, cfg["seed"]).score(uidx, cand_idx, train)
    return {"window":w,"users":users,"candidates":candidates,"old":old,"train":train,
            "gt":gt,"eligible":eligible,"gt_single":gt_single,"single_eligible":single_eligible,
            "pop":pop,"assoc":assoc,"repeat":repeat,"fallback":fallback,"cold":cold,
            "ppr":ppr,"opportunity":opportunity_factor(p,w,candidates),
            "n_qualified_train":len(qtrain)}


def fused(x, h, rw, aw, opportunity=False, ppr_weight=0.0):
    repeat = x["repeat"][h] * x["opportunity"][None,:] if opportunity else x["repeat"][h]
    scores = {"popularity":x["pop"],"repeat":repeat,"association":x["assoc"]}
    weights = {"popularity":1.0,"repeat":rw,"association":aw}
    norms = {"popularity":"global","repeat":"user","association":"user"}
    if ppr_weight:
        scores["pagerank"] = x["ppr"]
        weights["pagerank"] = ppr_weight
        norms["pagerank"] = "user"
    return fuse_scores(scores,weights,norms)


def evaluate_variant(x, name, scores, baseline_ap=None):
    recs, stats = recommend_from_scores(scores, x["users"], x["candidates"], x["old"],
                                        "allow", x["fallback"], cold_mask=x["cold"])
    allm = evaluate_recommendations(recs,x["gt"],x["users"],5,len(x["candidates"]))
    fivem = evaluate_recommendations(recs,x["gt"],x["eligible"],5,len(x["candidates"]))
    single = evaluate_recommendations(recs,x["gt_single"],x["single_eligible"],5,len(x["candidates"]))
    aps = np.array([ap_at_k(recs[int(u)],x["gt"].get(int(u),set()),5) or 0.0 for u in x["users"]], dtype=float)
    result = {"window":x["window"]["name"],"variant":name,
              "map5_all_users":allm["map5_all"],"map5_eligible5":fivem["map5_all"],
              "map5_single_eligible5":single["map5_all"],"n_eligible5":len(x["eligible"]),
              "fallback_used":stats["fallback_used"]}
    if baseline_ap is not None:
        delta = aps-baseline_ap
        result.update({"users_gained":int(np.count_nonzero(delta>1e-12)),
                       "users_lost":int(np.count_nonzero(delta< -1e-12)),
                       "users_tied":int(np.count_nonzero(abs(delta)<=1e-12)),
                       "mean_user_ap_delta":float(delta.mean())})
    return result,aps


def windows_for(phase, plan):
    cp = json.loads((ROOT/"corrected_plan.json").read_text(encoding="utf-8"))
    if phase == "diagnostic":
        return [cp["diagnostic_window"]]
    return cp["selection_windows"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("core","ablation","diagnostic"),required=True)
    args=parser.parse_args()
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    if sha256_file(resolve("midterm/submission_candidate_screen80.csv")) != plan["third_submitted_sha256"]:
        raise RuntimeError("third submitted CSV changed")
    if args.phase == "ablation":
        decision=json.loads((ROOT/"fourth_core_decision.json").read_text(encoding="utf-8"))
        if decision["core_csv_sha256"] != sha256_file(ROOT/"fourth_core.csv"):
            raise RuntimeError("core results changed")
    if args.phase == "diagnostic":
        decision=json.loads((ROOT/"fourth_selection_decision.json").read_text(encoding="utf-8"))
        if decision["ablation_csv_sha256"] != sha256_file(ROOT/"fourth_ablation.csv"):
            raise RuntimeError("ablation results changed")
    cfg=load_config();p=load_prepared(cfg)
    rows=[];stats=[]
    for w in windows_for(args.phase,plan):
        t=time.perf_counter()
        x=prepare_window(p,cfg,w,need_ppr=args.phase!="core")
        baseline_scores=fused(x,14.0,3.0,0.25)
        baseline, baseline_ap=evaluate_variant(x,"third_exact_config",baseline_scores)
        rows.append(baseline)
        if args.phase=="core":
            configs=itertools.product((7.0,14.0,28.0),(1.0,2.0,3.0,4.0),(0.0,0.25))
            for h,rw,aw in configs:
                name=config_id(h,rw,aw)
                if (h,rw,aw)==(14.0,3.0,0.25):
                    row=baseline.copy();row["variant"]=name
                else:
                    row,_=evaluate_variant(x,name,fused(x,h,rw,aw),baseline_ap)
                row.update({"half_life":h,"repeat_weight":rw,"association_weight":aw})
                rows.append(row)
        else:
            core=json.loads((ROOT/"fourth_core_decision.json").read_text(encoding="utf-8"))
            h,rw,aw=core["half_life"],core["repeat_weight"],core["association_weight"]
            variants={"selected_core":fused(x,h,rw,aw),
                      "opportunity":fused(x,h,rw,aw,opportunity=True),
                      "ppr_0p1":fused(x,h,rw,aw,ppr_weight=0.1),
                      "ppr_0p25":fused(x,h,rw,aw,ppr_weight=0.25)}
            if args.phase=="diagnostic":
                final=json.loads((ROOT/"fourth_selection_decision.json").read_text(encoding="utf-8"))["selected_variant"]
                variants={final:variants[final]}
            for name,scores in variants.items():
                row,_=evaluate_variant(x,name,scores,baseline_ap)
                rows.append(row)
        stats.append({"window":w["name"],"n_candidates":len(x["candidates"]),
                      "n_eligible5":len(x["eligible"]),"qualified_cold":int(x["cold"].sum()),
                      "n_qualified_train":x["n_qualified_train"],
                      "runtime_seconds":round(time.perf_counter()-t,3)})
        print(w["name"],"baseline",round(baseline["map5_eligible5"],5),"evaluated",len(rows),flush=True)
    stem={"core":"fourth_core","ablation":"fourth_ablation","diagnostic":"fourth_diagnostic"}[args.phase]
    pd.DataFrame(rows).to_csv(ROOT/f"{stem}.csv",index=False)
    (ROOT/f"{stem}_stats.json").write_text(json.dumps({"plan_sha256":sha256_file(PLAN),"windows":stats},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

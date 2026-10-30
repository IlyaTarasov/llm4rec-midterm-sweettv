"""Materialize a real frozen-model example for the author's manual check.

This performs an assistant-executed refit, not a new search or CSV export.
The answer is deliberately omitted from the evidence file.
"""
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from .config import load_config
from .evaluate import load_prepared, recommend_from_scores, fuse_scores
from .fourth_pass import prepare_window
from .eighth_pass import internal, history
from .corrected_target_experiment import ROOT

def main():
    cfg = load_config()
    prepared = load_prepared(cfg)
    window = json.loads((ROOT / 'corrected_plan.json').read_text())['selection_windows'][1]
    x = prepare_window(prepared, cfg, window)
    _, _, _, current = internal(prepared, window, x)
    count, age = history(x)
    pop = fuse_scores({'popularity': x['pop']}, {'popularity': 1.}, {'popularity': 'global'})
    score = pop + (count / (count + 5) * np.exp(-np.minimum(age, 1e9) / 56))[:, None] * (current - pop)
    recs, _ = recommend_from_scores(score, x['users'], x['candidates'], x['old'], 'allow', x['fallback'], cold_mask=x['cold'])
    matches = []
    for user in x['users']:
        user = int(user)
        target = x['gt'].get(user, set())
        if len(target) != 5:
            continue
        ranks = [rank for rank, item in enumerate(recs[user], 1) if item in target]
        if ranks == [2, 4]:
            matches.append(user)
    assert matches, 'No requested separated-hit example exists'
    user = min(matches)
    out = {'model': 'frozen confidence k5 tau56', 'window': window,
           'user_id': str(user), 'ordered_recommendation_ids': recs[user],
           'target_ids': sorted(x['gt'][user]),
           'target_order_note': 'Actual top-five frequency target, presented as a set in ascending ID order.',
           'executor': 'assistant generated inputs; author calculation pending',
           'evidence_basis': 'same frozen formula and window as report_evidence.json'}
    path = ROOT / 'manual_ap_inputs.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()

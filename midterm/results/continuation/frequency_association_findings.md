# Frequency ratios and association regularization: bounded follow-up

The submitted confidence/integrated-decay/MF CSV, SHA256 `4b8e43896f9f0cdd81d31427a2798969111cc59626d1ac8df86825a630d7077a`, is preserved with its config and manifest under `results/fifth_submitted/`. The user reports **private0.41770/public0.42724** for that upload. A new screenshot file was not supplied to this execution; the reported screenshot displays generic `submission.csv`, so file-to-score mapping is the user's statement. No screenshot inspection or independent Kaggle verification is claimed.

The assignment permits association rules and implicit count MF, and the existing corrected individual permitted-method/popularity comparisons and identical-two-user examples remain intact. The ledger review ruled out repeating the exhausted weight, half-life, rate, PPR, channel, MF and confidence grids. Poisson remains rejected under its original threshold. Two different hypotheses, formulas, resource bounds and a **0.003 mean eligible-five gain** guard were saved in `frequency_association_plan.json` before scoring. All comparisons preserve the submitted model's80%-qualified top5-frequency target, training-active population, complete-view cutoffs, users/candidates, confidence, repeat policy, fallback and uint64 handling. The submitted control exactly reproduces its frozen local selection metrics.

## Count-power result: rejected

Count powers0.5/1/1.5 replace `log1p(weighted_count)` while leaving integrated56-day decay, schedule opportunity and confidence fixed. This differs from the earlier failed completion-rate models: no exposure denominator or new smoothing parameters. All three lose on all-user and eligible-five MAP@5 in every selection window.

| Window | Submitted all-user/eligible-five | Power0.5 | Power1 | Power1.5 |
|---|---:|---:|---:|---:|
| Apr6–Jun29 | 0.41885/0.43914 | 0.41587/0.43563 | 0.40782/0.42720 | 0.39963/0.41874 |
| Apr13–Jul6 | 0.41475/0.43512 | 0.41277/0.43236 | 0.40472/0.42476 | 0.39552/0.41503 |
| Apr20–Jul13 | 0.41362/0.43187 | 0.40933/0.42703 | 0.39976/0.41729 | 0.39050/0.40802 |

Even power0.5 improves/harms only86/126,91/137,79/131 users in the three windows. Higher powers are worse. `frequency_count.json` retains every variant and past-defined history-count/recency cohort outcome. The count avenue was closed before association testing; the submitted log score stayed the control.

## Association result: insufficient gain

Baseline association edge weight is confidence after active-user lift filtering. Its Top5 item recommendations concentrate96.6%,97.4%,98.0% in each window's training-popularity top decile. This supports investigating popularity dominance but does not prove that those recommendations are irrelevant.

The four variants regularize existing eligible rule edges as `confidence * co_count/(co_count+lambda) * min((active_users/target_popularity)^beta,3)`, lambda10/50 × beta0.25/0.5. The same support/confidence/lift thresholds and confidence-top100 rule set are retained. No fusion-weight change or new family was used.

| Variant | Eligible-five deltas versus submitted, Apr6 / Apr13 / Apr20 |
|---|---|
| lambda10,beta0.25 | +0.000383 /−0.000361 /+0.000234 |
| lambda10,beta0.5 | +0.000133 /−0.000135 /+0.001416 |
| lambda50,beta0.25 | −0.000006 /+0.000178 /−0.000076 |
| lambda50,beta0.5 | +0.000488 /+0.000610 /+0.000938 |

The last variant also gains all-user MAP, but mean eligible-five gain is only about0.000679, below0.003. Its all-user/eligible-five values are0.41937/0.43963,0.41547/0.43574,0.41459/0.43281; paired gains/losses are57/35,51/42,67/44. Effects are not uniform across cohorts: the3–14-day recency cohort loses in the first window, and sparse1–4-event users are mixed. Full per-window/per-user-count and past-defined cohort summaries remain in `frequency_association.json`.

Neither hypothesis passes. Acceptance thresholds were not relaxed, no new diagnostic was run, and no CSV was created. Every temporal period/source has been inspected repeatedly and the windows overlap; no genuinely untouched holdout exists. Kaggle scores served only as provenance.

The strongest retained configuration remains the **submitted confidence-integrated-MF model**: log1p integrated qualifying-airing count with56-day decay, capped global opportunity correction smoothing0.2/exponent0.5, popularity1/association0.25/repeat2, retained implicit MF0.25, and confidence count/(count+5)*exp(−age/56). The new evidence supports stopping these avenues rather than wider neighboring grids. It does not establish a global optimum.

Technical checks: regularized-rule arithmetic, finite values and invariance to future-only indexed users have a synthetic test; full suite **34 tests passed**. Five submitted hashes and raw source hashes are checked in `frequency_association_manifest.json`. No upload, publication, PDF, session export, other-chat retrieval or subagent work occurred. Rejected experiments remain intact.

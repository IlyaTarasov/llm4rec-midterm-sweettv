# Final confidence and channel-exposure hypotheses

The four submitted files remain preserved. Best confirmed is the fourth opportunity strategy: **0.39382 private / 0.40756 public**. No additional submission CSV was generated.

Past-defined cohorts (`eighth_cohorts.json`) compare fourth versus the previous internal challenger using qualifying-history event counts and latest qualified-view age. Counts1–4 had more paired losses in two windows; stale profiles had more individual losses in all three, although mean differences were mixed. Those reused observations motivated a six-variant smooth shrinkage ablation, not a future-outcome-defined router. The formula/grid and stop rules were saved in `eighth_plan.json` before variant scoring.

The strongest supported configuration is the previous internal challenger (integrated event decay56 days, global opportunity smoothing0.2/exponent0.5/caps0.5–1.5, popularity1/association0.25/repeat2, qualified MF16/alpha5/reg0.05 contribution0.25), with personalization multiplied by

`qualified_history_count/(qualified_history_count+5) * exp(-latest_qualified_view_age_days/56)`.

The global popularity term stays at weight1. Existing zero-qualified-history and zero-score fallback remains unchanged. Both count and age use pre-cutoff complete qualifying views only. There is no learned router or future-user feature.

| Selection window | Fourth eligible-five MAP@5 | Prior internal | Confidence selected | Users gained/lost versus fourth |
|---|---:|---:|---:|---:|
| Apr6–Jun29 | 0.42919 | 0.43180 | **0.43914** | 271/181 |
| Apr13–Jul6 | 0.42696 | 0.43003 | **0.43512** | 278/198 |
| Apr20–Jul13 | 0.42217 | 0.42948 | **0.43187** | 291/208 |

All-user scores also improved in every selection window. Stronger shrinkage was inconsistent. All six comparisons, including failures, are in `eighth_confidence.json`.

The four personalized airing-exposure variants used training-only raw session/channel probabilities smoothed toward a training-only global channel prior, two profile shrinkages20/100 and two factor blends0.25/0.5. Future schedule information was limited to ID/channel/time/duration. Complete-airing counts were deduplicated and divided by the corresponding window length. Every variant lost eligible-five MAP@5 against the internal control in every selection window (roughly−0.0003 to−0.0022). This avenue was rejected; no grid expansion or combination with confidence was attempted. Results remain in `eighth_channel.json`.

The confidence choice was frozen in `eighth_final_decision.json` before one reused May4–July27 diagnostic:

| Strategy | All-user | Eligible-five | Single-view80 eligible-five |
|---|---:|---:|---:|
| Fourth submitted | 0.39097 | 0.40424 | 0.41011 |
| Prior internal | 0.39701 | 0.41104 | 0.41683 |
| Confidence selected | **0.39782** | **0.41222** | **0.41881** |

Confidence versus fourth gains user AP for313 users, loses for217, and ties for730. Its extra diagnostic gain over the previous internal model is small (+0.00118 eligible-five and +0.00081 all-user). Every period/source has been repeatedly inspected; temporal windows overlap. These are robustness observations, not independent confirmation or proof of a Kaggle gain. The fourth remains the confirmed choice and the new strategy remains internal.

The targeted synthetic test verifies that post-cutoff raw sessions cannot influence channel profiles and duplicate schedule airings cannot inflate exposure. Existing qualified-target, large-uint64, temporal-cutoff, fallback and submitted-file checks remain applicable. Provenance is saved in `eighth_manifest.json`. No intermediate CSV, upload, publication, report or session export was made. The two requested avenues are closed: channel personalization failed, and confidence has a modest supported gain with no justification for widening its grid. Further progress would require a new concrete hypothesis or independent evidence.

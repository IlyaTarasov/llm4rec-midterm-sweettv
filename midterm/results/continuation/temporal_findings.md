# Last temporal-availability hypothesis: rejected

The assignment allows past viewing context and future programme ID/channel/airtime availability. The ledger's early time experiment counted session starts and used superseded targets; it did not union viewing intervals, estimate duration-normalized temporal occupancy or replace the past/upcoming rate ratio. The present test was therefore valid and distinct, but remained auxiliary to the already evaluated permitted methods.

`temporal_plan.json` fixed four variants before scoring: hour24 or weekday/weekend-hour48 profiles, each blended into the incumbent global opportunity factor at0.25 or0.5. The criterion was unchanged: positive all-user and eligible-five gain in every selection window and mean eligible-five gain≥0.003. There was no genuinely untouched source/temporal holdout.

Complete valid past raw intervals, including unlabelled airtime, were unioned across channels per user. Overlapping or duplicate views could not inflate occupancy. Intervals were split at naive wall-clock hour/day boundaries, with no inferred timezone correction. Bin watched seconds were divided by actual available seconds in the fit window, and smoothed with7 hours of bin exposure at the training-active-user global occupancy prior. Empty profiles received that prior; recommendation cold/zero-score fallback stayed unchanged.

Distinct labelled complete schedule airings were split across the same bins. Each airing contributed its mean temporal occupancy over its full duration; weighted airing counts were divided by the appropriate past/future window days. Rates were capped2/day and the same fixed smoothing0.2/square-root ratio/caps0.5–1.5 were applied. Blending affected only the integrated-repeat opportunity term. MF, confidence, global popularity, association, users/candidates/repeats and qualified80/top5-frequency target remained fixed. Upcoming programme titles/categories and future views were never used.

Eligible-five MAP@5 deltas versus submitted confidence control:

| Variant | Apr6–Jun29 | Apr13–Jul6 | Apr20–Jul13 |
|---|---:|---:|---:|
| hour24,blend0.25 | +0.000026 | +0.000542 | −0.000501 |
| hour24,blend0.5 | −0.000771 | −0.000390 | −0.001188 |
| weekday/weekend48,blend0.25 | −0.000125 | +0.000496 | −0.000484 |
| weekday/weekend48,blend0.5 | −0.001026 | −0.000737 | −0.001044 |

The best tested mean gain is only **0.000022**, with inconsistent window direction. Its all-user/eligible-five scores are0.41890/0.43917,0.41531/0.43567,0.41311/0.43137, versus control0.41885/0.43914,0.41475/0.43512,0.41362/0.43187. Paired user gains/losses are25/25,28/22,24/28. Past-defined count/recency cohort summaries are retained for every variant in `temporal_selection.json`; they are reused development diagnostics and do not establish a stable benefit. No variant passed, no threshold was weakened, and no final diagnostic or single-view rescore was run.

The implementation completed each window in22.4–24.7 seconds, below its180-second budget. Tests verify cross-hour/midnight/weekend splitting, union of duplicate raw views, exclusion of crossing-cutoff views, finite training-prior empty-history behavior and duplicate-airing invariance. The full suite has **36 passing tests**. The exact submitted control reproduces frozen selection scores, and all five submitted hashes remain unchanged (`temporal_manifest.json`). No new CSV was created.

This round is closed. The strongest retained configuration remains the submitted confidence-integrated-MF model; the user reports **0.41770 private /0.42724 public** for its hash, with the generic-screenshot-name caveat recorded separately. Count powers, regularized associations and now temporal availability failed their declared retention rules. The evidence supports stopping these avenues rather than expanding nearby grids, without claiming a global optimum. All periods were repeatedly inspected and overlap. No upload, report PDF, publication, session export, other-chat retrieval or subagent work occurred.

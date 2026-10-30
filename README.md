# SWEET.TV programme recommendation — Ilya Tarasov

Implementation, tests, configuration and experiment evidence for the
individual SWEET.TV midterm. Repository:
https://github.com/IlyaTarasov/llm4rec-midterm-sweettv

The designated fifth file is `midterm/submission_final_confidence_integrated_mf.csv`,
SHA256 `4b8e43896f9f0cdd81d31427a2798969111cc59626d1ac8df86825a630d7077a`.
The author uploaded it and supplied the returned private/public scores
**0.41770 / 0.42724**. The generic screenshot filename has no content hash;
binding the score to this artifact relies on the author's upload record.
Search is closed. Other root submission CSVs are immutable historical outputs,
included because the original reproduction checks assert their hashes.

## Reproduce

Use Python 3.13.0 with `requirements.txt` (NumPy 2.3.3, pandas 2.3.3,
SciPy 1.18.1). The recorded machine was Windows 11 with Intel i7-12700H.
Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before
starting Python. Commands below run from this payload's root.

Download the competition data yourself under its terms from
https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data .
Place `dataset11-30.csv`, `export_arh_11-20-final.csv`,
`export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` and the original
`submission.csv` template in `midterm/data/sweettv-tv-program-recommender/`.
These licensed inputs, derived input caches and raw-view extracts are excluded.

```
python -m midterm.src.pipeline --stage prepare
python -m unittest discover -s midterm/tests -v
python -m midterm.src.final_confidence_candidate --verify-existing
python -m midterm.src.verify_final_confidence_csv
```

The final refit verifies the existing designated file byte for byte; it does
not create an intermediate submission CSV. It refreshes an audit manifest.
The 60-second default is used for historical preparation and shared fallback;
Fallback distinct-user counts use overlaps >=60 seconds with overlap_end
<= cutoff, rather than the personal model's complete source-view/airing rule.
The ordinary popularity comparison uses >=80%-qualified airings, so a cold
user's common fallback can differ from a warm user's popularity list. This
same fallback policy is applied to every method. The corrected personal
training/targets are constructed by the final entry point
using 80%-qualified airings and top-five frequency targets. Do not use the
historical `pipeline --stage final` to reproduce the designated fifth model.

For the report's common April–July comparison:
```
python -c "from pathlib import Path; Path('midterm/artifacts/seventh').mkdir(parents=True, exist_ok=True)"
python -m midterm.src.report_evidence
```
This writes comparison JSON and an ignored MF cache, not a submission CSV.
Individual methods share users, candidates, repeats, target and cold fallback.
The PPR result is a capped numerical approximation; no convergence residual
was retained for this comparison.

## Evidence and status

`midterm/CURRENT_STATUS.md` is authoritative. Older “latest”, “unsubmitted”
and “current” wording in historical records is superseded by it. The selection
periods overlap and were reused; there is no untouched final temporal holdout.
Late count-power, regularized-association and temporal experiments failed
their declared retention criteria. No global optimum or product lift is claimed.

The original model/config/test files are copied byte for byte. Evidence JSON
and Markdown with host paths has portable projections at the same relative
paths; `publication_manifest.json` maps each to the untouched original hash
and states the transformation. Metrics, source hashes and rankings are not
changed. The manifest is a projection audit, not a claim that projected JSON
is a native session export. Superseded raw-view extracts, technical-session
exports, credentials, private host paths, course documents, unrelated homework,
caches and generated reports are excluded from this payload.

OpenCode/Codex assisted implementation and checking; the author's separate
manual-verification evidence and genuine native technical exports belong in
the local course review package, not in this public code payload.

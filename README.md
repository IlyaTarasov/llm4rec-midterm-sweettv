# SWEET.TV programme recommendation

Predict five programme IDs for each of 1,260 viewers. The designated final
submission returned **MAP@5 0.41770 private / 0.42724 public** on Kaggle.

```text
config.json                              Frozen final settings and source hashes
requirements.txt                         Pinned numerical dependencies
sweettv/
  data.py                                Source readers, interval join, qualification
  models.py                              Popularity, association rules, implicit MF
  ranking.py                             Normalization, ranking and shared fallback
  pipeline.py                            Complete raw-data refit and verification
  comparison.py                          Content/PPR and fixed five-method comparison
submission_final_confidence_integrated_mf.csv
```

## Reproduce and verify

Use Python 3.13.0. The reference environment is Windows 11, Intel i7-12700H,
NumPy 2.3.3, pandas 2.3.3 and SciPy 1.18.1. The entry point sets BLAS thread
counts to one before importing NumPy.

Download the licensed inputs from the [competition data page](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data)
under its terms. Place these five files in `data/`:

```text
data/dataset11-30.csv
data/export_arh_11-20-final.csv
data/export_arh_21-30-final.csv
data/export_arh_31-42-final.csv
data/submission.csv                       Original competition template
```

From the project root:

```shell
python -m pip install -r requirements.txt
python -m sweettv.pipeline --data-dir data --verify
```

Verification checks source hashes, joins the raw views and schedules, trains
the models from scratch, regenerates the submission in memory and compares
every byte against the designated CSV. It needs no historical outputs, result
JSON, prepared caches or files outside this project and the five source CSVs.
It does not overwrite the submitted file or produce intermediate CSVs.

To reproduce the fixed April–July comparison of content similarity, association,
implicit MF, personalized PageRank, popularity and the final fusion:

```shell
python -m sweettv.comparison --data-dir data
python -m sweettv.comparison --data-dir data --user-id 13571199046623737884 --user-id 17337543921140631187
```

The second command also prints ordered programme IDs, titles, categories and
held-out outcomes for the same two example users. These commands print JSON
to the terminal without writing result files or generating CSV candidates.
The comparison settings are explicit under `comparison` in `config.json`.
Content uses training-only TF–IDF metadata and 14-day history decay. PPR uses
top-50 cosine co-view edges, damping 0.95 and 30-day restart-history decay,
with at most 30 iterations and tolerance 1e-7; convergence is not assumed.

Expected SHA256:
`4b8e43896f9f0cdd81d31427a2798969111cc59626d1ac8df86825a630d7077a`

## Model

The configured `fit_cutoff` of 27 July 2020 is the historical viewing-log
boundary used to fit and score the recommender, not the date this project was
trained. The fitting window is [9 March, 27 July 2020): airings must start
before 00:00 on 27 July, and complete source views and airings may end exactly
at that boundary. The future schedule supplies candidate programme IDs,
channels, airing start times and durations for [27 July, 20 October 2020),
including the availability adjustment; future user viewing behavior is not
used. Overlapping view fragments are unioned per user/airing, and an airing
qualifies at 80% screen time. Personal training requires complete source views
and airings by the cutoff. User IDs remain uint64.

The implemented methods are distinct-user popularity, co-view association
rules (support 10, confidence 0.05, active-user lift 1, top 100 neighbors) and
implicit ALS matrix factorization (16 factors, alpha 5, regularization 0.05,
15 iterations, seed 42). The final fusion is
`P + g * (0.25*A + 2*R + 0.25*M)`, with per-user normalization of personal
scores and globally normalized popularity. Repeat counts decay with a 56-day
half-life and receive a capped future/past airing-rate adjustment.
`g = n/(n+5) * exp(-age/56)` shrinks weak or stale history toward popularity.
All exact settings are in `config.json`.

Repeats are allowed; equal scores prefer the smaller programme ID. Cold users
and unusable scores share distinct-user popularity from training overlaps of
at least 60 seconds, whose overlap ends by cutoff. This historical fallback
is identical across methods and differs from the main 80%-qualified baseline.

## Limitations

Programme labels are sparse; missing viewing is not dislike. Frequency favors
familiar programmes and schedule exposure. Validation windows were reused
and overlap, so the scores do not establish an untouched-holdout estimate,
user satisfaction, discovery quality or an online business effect. Future
viewing is never used; future schedule features are restricted to programme
ID, channel and timing. Raw data, caches and reports are excluded from Git.

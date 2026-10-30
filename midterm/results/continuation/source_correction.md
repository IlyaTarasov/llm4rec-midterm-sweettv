# Verified competition target description, 2026-10-30

The fully rendered [official Kaggle Data page](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), under **Dataset Description**, was read directly in the browser. It states:

- `vsetv_id` equals the schedule's `channel_id`.
- `tv_show_id=0` schedule rows are dropped.
- Screen time is watched time divided by programme duration; only screen time **at least 0.8** is counted.
- The five most frequent high-screen-time programmes are selected for each user, in descending frequency. The solution has exactly five per user, and the public/private split is **by users, 50/50**.
- The test period is the next 12 weeks (weeks 31–42).

This corrects the earlier statement that hidden target construction was undocumented. The earlier 60-second, all-viewed-ID proxy and its five sensitivity proxies remain reproducible historical diagnostics, but they are not the documented optimization target. Their ranking conclusions should not drive a new submission.

The page does **not** specify how multiple viewing fragments of one airing are combined, how ties in programme frequency are broken, or how the private implementation computes MAP from an ordered five-ID solution. A local reconstruction must state its choices and limits. No Kaggle labels or other entrants' solutions have been accessed.

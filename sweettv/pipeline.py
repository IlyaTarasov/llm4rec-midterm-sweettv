"""Reproduce the frozen final submission directly from the five source CSVs."""
from __future__ import annotations

import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .data import (SCHEDULE_FILES, read_schedule, read_views, read_submission,
                   join_views_to_schedule, candidate_items, qualify_airings,
                   build_train_data)
from .models import PopularityModel, AssociationModel, MFModel
from .ranking import fuse_scores, recommend_from_scores

PROJECT = Path(__file__).resolve().parents[1]
SUBMISSION = 'submission_final_confidence_integrated_mf.csv'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(data_dir, config):
    """Read raw data only; do not load or create any prepared cache."""
    for name, expected in config['source_sha256'].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f'Source SHA256 mismatch: {name}')
    schedules = {name: read_schedule(data_dir / name) for name in SCHEDULE_FILES}
    views = read_views(data_dir / 'dataset11-30.csv')
    template = read_submission(data_dir / 'submission.csv')
    past = pd.concat([schedules[name] for name in SCHEDULE_FILES[:2]], ignore_index=True)
    past = past.drop_duplicates(['channel_id', 'start', 'duration', 'tv_show_id'], keep='first')
    joined = join_views_to_schedule(views, past, min_overlap_seconds=0,
                                    min_overlap_fraction=0., best_overlap_only=False)
    users = np.sort(views['user_id'].astype('uint64').unique())
    items = np.sort(np.unique(np.concatenate([
        s.loc[s.tv_show_id != 0, 'tv_show_id'].astype('int64').to_numpy()
        for s in schedules.values()
    ])))
    return schedules, joined, users, items, template


def complete_airings(schedule, lo, hi):
    return schedule.loc[(schedule.tv_show_id != 0) & (schedule.start >= lo)
                        & (schedule.start < hi) & (schedule.end <= hi),
                        ['tv_show_id', 'channel_id', 'start', 'duration', 'end']].drop_duplicates()


def opportunity(past, future, start, cutoff, end, candidates, smoothing, config):
    def rate(schedule, lo, hi):
        count = complete_airings(schedule, lo, hi).groupby('tv_show_id').size()
        count = count.reindex(candidates, fill_value=0).to_numpy(dtype=float)
        return np.minimum(count / ((hi - lo).total_seconds() / 86400),
                          config['daily_rate_cap'])
    ratio = ((rate(future, cutoff, end) + smoothing) /
             (rate(past, start, cutoff) + smoothing))
    return np.clip(ratio ** config['exponent'], config['minimum'], config['maximum'])


def refit(schedules, joined, user_ids, item_ids, template, config):
    start, cutoff, end = map(pd.Timestamp, (config['fit_start'], config['fit_cutoff'],
                                          config['candidate_end_exclusive']))
    users = template.user_id.to_numpy(dtype=np.uint64)
    future = schedules[SCHEDULE_FILES[2]][['tv_show_id', 'channel_id', 'start', 'duration', 'end']]
    candidates = candidate_items(future, cutoff, end)
    uidx = np.searchsorted(user_ids, users)
    cidx = np.searchsorted(item_ids, candidates)
    qualified = qualify_airings(joined, start, cutoff)
    pseudo = qualified[['user_id', 'item_id', 'watched_seconds', 'screen_time',
                        'last_view', 'sched_end']].copy()
    pseudo.rename(columns={'watched_seconds': 'overlap_seconds', 'screen_time': 'overlap_fraction',
                           'last_view': 'view_start', 'sched_end': 'overlap_end'}, inplace=True)
    cfg = {'join': {'min_overlap_seconds': 0, 'min_overlap_fraction': 0.},
           'seed': config['seed'], 'popularity': {'weight': 'users'},
           'association': config['association'],
           'mf': {**config['implicit_mf'], 'weight': 'count'}}
    train = build_train_data(cfg, pseudo, user_ids, item_ids, pd.DataFrame(), cutoff)
    fallback_cfg = {**cfg, 'join': {'min_overlap_seconds': config['fallback_min_overlap_seconds'],
                                  'min_overlap_fraction': 0.}}
    old = build_train_data(fallback_cfg, joined, user_ids, item_ids, pd.DataFrame(), cutoff)
    seed = config['seed']
    pop = PopularityModel().fit(train, cfg, seed).score(uidx, cidx, train)
    fallback = PopularityModel().fit(old, cfg, seed).score(uidx, cidx, old)[0]
    association = AssociationModel().fit(train, cfg, seed).score(uidx, cidx, train)

    # Sum decay separately for every qualified airing, preserving uint64 users.
    order = np.argsort(users)
    sorted_users = users[order]
    uid = qualified.user_id.to_numpy(dtype=np.uint64)
    iid = qualified.item_id.to_numpy(dtype=np.int64)
    upos, ipos = np.searchsorted(sorted_users, uid), np.searchsorted(candidates, iid)
    valid = (upos < len(users)) & (ipos < len(candidates))
    valid &= sorted_users[np.clip(upos, 0, len(users)-1)] == uid
    valid &= candidates[np.clip(ipos, 0, len(candidates)-1)] == iid
    ages = (cutoff - qualified.loc[valid, 'sched_end']).dt.total_seconds().to_numpy() / 86400.
    weighted = np.zeros((len(users), len(candidates)), dtype=np.float64)
    np.add.at(weighted, (order[upos[valid]], ipos[valid]),
              np.power(.5, np.maximum(ages, 0) / config['repeat']['half_life_days']))
    past = pd.concat([schedules[name] for name in SCHEDULE_FILES[:2]], ignore_index=True)
    ocfg = config['opportunity']
    # Keep the original multiplication/division order before the selected factor,
    # so this packaging refactor also preserves floating-point tie behavior.
    roundtrip = opportunity(past, future, start, cutoff, end, candidates, .1, ocfg)
    repeat = (np.log1p(weighted) * roundtrip) / roundtrip
    repeat *= opportunity(past, future, start, cutoff, end, candidates,
                          ocfg['smoothing'], ocfg)[None, :]
    current = fuse_scores({'popularity': pop, 'association': association, 'repeat': repeat},
                          {'popularity': config['popularity']['fusion_weight'],
                           'association': config['association']['fusion_weight'],
                           'repeat': config['repeat']['fusion_weight']},
                          {'popularity': 'global', 'association': 'user', 'repeat': 'user'})
    mf = MFModel().fit(train, cfg, seed).score(uidx, cidx, train)
    maximum = mf.max(axis=1, keepdims=True)
    mf /= np.where(maximum > 0, maximum, 1)
    current += config['implicit_mf']['fusion_weight'] * mf
    interaction = train.interactions
    count = interaction.groupby('user_id').view_count.sum().reindex(users, fill_value=0).to_numpy(float)
    last = interaction.groupby('user_id').last_view.max().reindex(users)
    age = (cutoff - last).dt.total_seconds().to_numpy() / 86400.
    age = np.where(np.isfinite(age), age, np.inf)
    ccfg = config['confidence']
    confidence = count / (count + ccfg['count_shrinkage']) * np.exp(-np.minimum(age, 1e9) / ccfg['age_decay_days'])
    global_pop = fuse_scores({'popularity': pop}, {'popularity': 1.}, {'popularity': 'global'})
    scores = global_pop + confidence[:, None] * (current - global_pop)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite final scores')
    cold = np.diff(train.X_binary.indptr)[uidx] == 0
    recs, stats = recommend_from_scores(scores, users, candidates, old, 'allow', fallback, cold_mask=cold)
    return recs, candidates, stats, len(qualified)


def serialize(recs, candidates, data_dir):
    with (data_dir / 'submission.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if fields != ['user_id', 'tv_show_id'] or len({r['user_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid template schema/user IDs')
    allowed = set(map(int, candidates))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        items = recs[int(row['user_id'])]
        if len(items) != 5 or len(set(items)) != 5 or any(i <= 0 or i not in allowed for i in items):
            raise ValueError('Invalid recommendation list')
        writer.writerow({'user_id': row['user_id'], 'tv_show_id': ' '.join(map(str, items))})
    return buffer.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.json')
    parser.add_argument('--verify', action='store_true', help='Refit from raw data and compare the existing final CSV byte for byte')
    args = parser.parse_args()
    if not args.verify:
        parser.error('Use --verify to reproduce the final file in memory without overwriting it')
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print('Reading and joining raw source files (no prepared caches)...', flush=True)
    prepared = prepare(args.data_dir.resolve(), config)
    print('Fitting popularity, association, implicit MF and final fusion...', flush=True)
    recs, candidates, stats, qualified = refit(*prepared, config)
    generated = serialize(recs, candidates, args.data_dir.resolve())
    digest = hashlib.sha256(generated).hexdigest()
    if digest != config['expected_submission_sha256'] or generated != (PROJECT / SUBMISSION).read_bytes():
        raise ValueError('Regenerated final CSV differs from the designated artifact')
    print(json.dumps({'byte_identical': True, 'sha256': digest, 'bytes': len(generated),
                      'users': len(recs), 'candidates': len(candidates), 'qualified_airings': qualified,
                      'fallback': stats, 'seconds': round(time.perf_counter()-started, 3)}, indent=2))


if __name__ == '__main__':
    main()

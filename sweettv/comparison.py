"""Fixed temporal comparison of all reported methods, without model selection."""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name]="1"
import argparse,json,math,re,time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from .models import BaseModel, TrainData, PopularityModel, AssociationModel, MFModel
from .data import SCHEDULE_FILES, qualify_airings, candidate_items, build_train_data
from .ranking import recommend_from_scores
from .pipeline import prepare, refit, PROJECT

def parse_year(value: str) -> float:
    if not isinstance(value, str):
        return float("nan")
    m = re.search(r"((?:19|20)\d{2})", value)
    return float(m.group(1)) if m else float("nan")

def _first_nonempty(values: Iterable[str]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def build_item_metadata(schedules: Sequence[pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    """Aggregate programme attributes from schedule rows strictly before cutoff.

    Only the programme/channel/airtime columns are used for candidate
    availability elsewhere.  Genres, cast, director, year, title and category
    here are training-time metadata; no post-cutoff schedule metadata is read.
    """
    cols = [
        "tv_show_id", "start", "duration", "tv_show_title", "tv_show_category",
        "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
        "year_of_production", "director", "actors", "channel_id",
    ]
    frames = []
    for s in schedules:
        x = s.loc[(s["tv_show_id"] != 0) & (s["start"] < cutoff), cols]
        frames.append(x)
    if not frames:
        return pd.DataFrame(columns=["title", "category", "genres", "year", "director",
                                     "actors", "channels", "n_airings"])
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows.drop_duplicates(
        subset=["channel_id", "start", "duration", "tv_show_id"], keep="first"
    )

    recs = []
    for item, g in allrows.groupby("tv_show_id", sort=True):
        title_counts = g["tv_show_title"].value_counts()
        cat_counts = g["tv_show_category"].value_counts()
        genres = set()
        for col in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
            for v in g[col].unique():
                if isinstance(v, str) and v.strip():
                    genres.add(v.strip())
        years = pd.Series([parse_year(v) for v in g["year_of_production"].unique()]).dropna()
        year = float(years.mode().iloc[0]) if len(years) else float("nan")
        recs.append({
            "item_id": int(item),
            "title": title_counts.index[0] if len(title_counts) else "",
            "category": cat_counts.index[0] if len(cat_counts) else "",
            "genres": sorted(genres),
            "year": year,
            "director": _first_nonempty(g["director"].unique()),
            "actors": _first_nonempty(g["actors"].unique()),
            "channels": sorted({int(c) for c in g["channel_id"].unique()}),
            "n_airings": int(len(g)),
        })
    if not recs:
        return pd.DataFrame(columns=["title", "category", "genres", "year", "director",
                                     "actors", "channels", "n_airings"])
    return pd.DataFrame(recs).set_index("item_id").sort_index()

class ContentModel(BaseModel):
    name = "content"

    def _tokens(self, row: pd.Series, cfg: Dict) -> List[Tuple[str, float]]:
        c = cfg.get("content", {})
        out: List[Tuple[str, float]] = []
        cat = str(row.get("category", "") or "").strip()
        if cat:
            out.append((f"cat:{cat}", 1.0))
        genres = row.get("genres", [])
        if isinstance(genres, list):
            for g in genres:
                if str(g).strip():
                    out.append((f"genre:{str(g).strip()}", 1.0))
        year = row.get("year", np.nan)
        if pd.notna(year):
            out.append((f"decade:{int(year) // 10 * 10}s", float(c.get("decade_weight", 0.5))))
        if c.get("use_cast", True):
            director = str(row.get("director", "") or "").strip()
            if director:
                for name in director.split(","):
                    if name.strip():
                        out.append((f"dir:{name.strip().lower()}", float(c.get("director_weight", 0.5))))
            actors = str(row.get("actors", "") or "").strip()
            if actors:
                for name in actors.split(","):
                    if name.strip():
                        out.append((f"actor:{name.strip().lower()}", float(c.get("cast_weight", 0.35))))
        return out

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "ContentModel":
        c = cfg.get("content", {})
        n_items = data.n_items
        meta = data.item_meta.reindex(pd.Index(data.item_ids))
        token_lists: List[List[Tuple[str, float]]] = []
        df_counts: Dict[str, int] = {}
        for item in data.item_ids:
            row = meta.loc[item] if item in meta.index else pd.Series(dtype=object)
            toks = self._tokens(row, cfg)
            token_lists.append(toks)
            for tok, _ in toks:
                df_counts[tok] = df_counts.get(tok, 0) + 1
        min_df = int(c.get("min_token_df", 2))
        vocab = {tok: j for j, tok in enumerate(sorted(t for t, n in df_counts.items() if n >= min_df))}
        n_feat = len(vocab)
        # IDF is fitted on training-time metadata documents only.  The global
        # item array also contains future candidate ids, but they contribute no
        # metadata documents and must not change the weights.
        n_docs = max(len(data.item_meta), 1)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i, toks in enumerate(token_lists):
            per_item: Dict[int, float] = {}
            for tok, type_weight in toks:
                j = vocab.get(tok)
                if j is None:
                    continue
                idf = math.log((1.0 + n_docs) / (1.0 + df_counts[tok])) + 1.0
                per_item[j] = per_item.get(j, 0.0) + type_weight * idf
            norm = math.sqrt(sum(v * v for v in per_item.values()))
            if norm > 0:
                for j, v in per_item.items():
                    rows.append(i)
                    cols.append(j)
                    vals.append(v / norm)
        self.X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, max(n_feat, 1)))
        self.meta = meta
        self.cfg = cfg
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        half_life = float(self.cfg.get("content", {}).get("half_life_days", 30.0))
        Xc = self.X[cand_idx]
        out = np.zeros((len(uidx), len(cand_idx)), dtype=np.float64)
        for r, u in enumerate(uidx):
            idx, counts = data.history_row(int(u))
            if len(idx) == 0:
                continue
            w = data.recency_weights(int(u), half_life)
            profile = self.X[idx].T.dot(w)
            n = float(np.linalg.norm(profile))
            if n <= 0:
                continue
            out[r] = Xc.dot(profile / n)
        return out

def personalized_pagerank(
    PT: sparse.csr_matrix,
    e: np.ndarray,
    alpha: float,
    dangling: np.ndarray,
    iterations: int = 100,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, float, int]:
    """Personalized PageRank with dangling mass returned to the restart vector.

    ``PT`` is the transpose of a row-stochastic transition matrix whose dangling
    rows are all zero.  For a column vector ``p``, ``PT @ p`` therefore has mass
    ``sum(p) - sum(p[dangling])``; the missing mass is added back to ``e``.
    """
    e = np.asarray(e, dtype=np.float64).copy()
    s = e.sum()
    if s <= 0:
        return e, 0.0, 0
    e /= s
    p = e.copy()
    mass_err = 0.0
    used = 0
    for used in range(1, iterations + 1):
        dm = float(p[dangling].sum()) if bool(np.any(dangling)) else 0.0
        new = alpha * (PT @ p) + (1.0 - alpha + alpha * dm) * e
        mass_err = float(abs(new.sum() - 1.0))
        if np.abs(new - p).sum() < tol:
            p = new
            break
        p = new
    return p, mass_err, used

class PageRankModel(BaseModel):
    name = "pagerank"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "PageRankModel":
        c = cfg.get("pagerank", {})
        self.cfg = cfg
        Xb = data.X_binary
        n_items = data.n_items
        pop = np.asarray(Xb.sum(axis=0)).ravel().astype(float)
        co = (Xb.T @ Xb).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        if c.get("edge_weight", "cosine") == "cosine":
            d = np.sqrt(np.maximum(pop, 1e-12))
            co = sparse.diags(1.0 / d).dot(co).dot(sparse.diags(1.0 / d)).tocsr()
        top_k = int(c.get("top_k", 50))
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i in range(n_items):
            s, e = co.indptr[i], co.indptr[i + 1]
            js = co.indices[s:e]
            w = co.data[s:e]
            if len(js) > top_k:
                order = np.lexsort((js, -w))[:top_k]
                js, w = js[order], w[order]
            rows.extend([i] * len(js))
            cols.extend(js.tolist())
            vals.extend(w.tolist())
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        rowsum = np.asarray(W.sum(axis=1)).ravel()
        self.dangling = rowsum <= 0
        inv = np.zeros_like(rowsum, dtype=np.float64)
        np.divide(1.0, rowsum, out=inv, where=~self.dangling)
        P = (sparse.diags(inv).dot(W)).tocsr()  # row-stochastic except dangling rows
        self.PT = P.T.tocsr()
        self.n_items = n_items
        self.pop = pop
        self.last_mass_error = 0.0
        self.last_iterations = 0
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        c = self.cfg.get("pagerank", {})
        alpha = float(c.get("alpha", 0.85))
        iters = int(c.get("iterations", 30))
        tol = float(c.get("tol", 1e-7))
        half_life = float(self.cfg.get("content", {}).get("half_life_days", 30.0))
        out = np.zeros((len(uidx), len(cand_idx)), dtype=np.float64)
        for r, u in enumerate(uidx):
            idx, counts = data.history_row(int(u))
            if len(idx) == 0:
                continue
            w = data.recency_weights(int(u), half_life)
            e = np.zeros(self.n_items, dtype=np.float64)
            e[idx] = w
            s = e.sum()
            if s <= 0:
                continue
            e /= s
            p, mass_err, used = personalized_pagerank(
                self.PT, e, alpha, self.dangling, iterations=iters, tol=tol
            )
            self.last_mass_error = max(self.last_mass_error, mass_err)
            self.last_iterations = max(self.last_iterations, used)
            out[r] = p[cand_idx]
        return out

def top5_frequency(qualified: pd.DataFrame, candidates: np.ndarray | None = None) -> tuple[dict[int, set[int]], pd.DataFrame]:
    """Count qualified airings by programme and retain at most five per user."""
    q = qualified
    if candidates is not None:
        q = q.loc[q["item_id"].isin(set(map(int, candidates)))]
    freq = q.groupby(["user_id", "item_id"], sort=False).size().rename("frequency").reset_index()
    freq = freq.sort_values(["user_id", "frequency", "item_id"], ascending=[True, False, True], kind="mergesort")
    top = freq.groupby("user_id", sort=False).head(5)
    gt = {int(user): set(map(int, rows["item_id"])) for user, rows in top.groupby("user_id", sort=False)}
    return gt, top

def ap_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 5) -> float | None:
    """Average precision at k. Returns None when there is no relevant item."""
    if not relevant:
        return None
    hits = 0
    total = 0.0
    seen = set()
    for rank, item in enumerate(recommended[:k], start=1):
        item = int(item)
        # A repeated id is invalid as a second recommendation.  It consumes the
        # rank but cannot count as another hit; this prevents duplicate
        # predictions from inflating AP.
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def compare(data_dir, config, example_users=()):
    schedules, joined, user_ids, item_ids, template = prepare(data_dir, config)
    settings = config['comparison']
    local = {**config, 'fit_cutoff': settings['fit_cutoff'],
             'candidate_end_exclusive': settings['candidate_end_exclusive']}
    start, cutoff, end = map(pd.Timestamp, (local['fit_start'], local['fit_cutoff'],
                                          local['candidate_end_exclusive']))
    schedule = pd.concat([schedules[name] for name in SCHEDULE_FILES[:2]], ignore_index=True)
    candidates = candidate_items(schedule, cutoff, end)
    users = template.user_id.to_numpy(dtype=np.uint64)
    uidx, cidx = np.searchsorted(user_ids, users), np.searchsorted(item_ids, candidates)
    qualified = qualify_airings(joined, start, cutoff)
    pseudo = qualified[['user_id','item_id','watched_seconds','screen_time','last_view','sched_end']].copy()
    pseudo.rename(columns={'watched_seconds':'overlap_seconds','screen_time':'overlap_fraction',
                           'last_view':'view_start','sched_end':'overlap_end'}, inplace=True)
    cfg = {'join':{'min_overlap_seconds':0,'min_overlap_fraction':0.},'seed':config['seed'],
           'popularity':{'weight':'users'},'association':config['association'],
           'mf':{**config['implicit_mf'],'weight':'count'},'content':settings['content'],
           'pagerank':settings['pagerank']}
    metadata = build_item_metadata([schedules[name] for name in SCHEDULE_FILES[:2]], cutoff)
    train = build_train_data(cfg,pseudo,user_ids,item_ids,metadata,cutoff)
    oldcfg = {**cfg,'join':{'min_overlap_seconds':config['fallback_min_overlap_seconds'],
                          'min_overlap_fraction':0.}}
    old = build_train_data(oldcfg,joined,user_ids,item_ids,pd.DataFrame(),cutoff)
    fallback = PopularityModel().fit(old,cfg,config['seed']).score(uidx,cidx,old)[0]
    cold = np.diff(train.X_binary.indptr)[uidx] == 0
    gt, _ = top5_frequency(qualify_airings(joined,cutoff,end), candidates)
    eligible = np.array([len(gt.get(int(u),set()))==5 for u in users])
    display = pd.concat(list(schedules.values()),ignore_index=True).drop_duplicates('tv_show_id').set_index('tv_show_id')
    method_results, examples = [], []

    def collect(name,recs,elapsed=None):
        aps = np.array([ap_at_k(recs[int(u)],gt.get(int(u),set())) or 0. for u in users])
        method_results.append({'method':name,'map5_all_users':float(aps.mean()),
                               'map5_eligible5':float(aps[eligible].mean()),
                               'fit_score_seconds':elapsed})
        for user in example_users:
            if user not in recs:raise ValueError('Requested example user is not in template')
            seen=set(train.interactions.loc[train.interactions.user_id==user,'item_id'])
            for rank,item in enumerate(recs[user],1):
                row=display.loc[item]
                examples.append({'user_id':str(user),'method':name,'rank':rank,'id':item,
                                 'title':row.tv_show_title,'category':row.tv_show_category,
                                 'hit':item in gt.get(user,set()),'seen_training':item in seen,
                                 'target_ids':sorted(gt.get(user,set())),
                                 'ap5':ap_at_k(recs[user],gt.get(user,set()))})

    for name,cls in [('Popularity',PopularityModel),('Association',AssociationModel),
                     ('Content',ContentModel),('MF',MFModel),('PPR',PageRankModel)]:
        method_cfg=cfg
        if name=='PPR':
            method_cfg={**cfg,'content':{**cfg['content'],'half_life_days':settings['pagerank_history_half_life_days']}}
        t=time.perf_counter()
        model=cls().fit(train,method_cfg,config['seed'])
        scores=model.score(uidx,cidx,train)
        elapsed=time.perf_counter()-t
        recs,_=recommend_from_scores(scores,users,candidates,old,'allow',fallback,cold_mask=cold)
        collect(name,recs,elapsed)
        del scores
    recs,_,_,_=refit(schedules,joined,user_ids,item_ids,template,local,evaluation_schedule=schedule)
    collect('Final',recs)
    return {'fit_start':str(start.date()),'cutoff':str(cutoff.date()),'end_exclusive':str(end.date()),
            'users':len(users),'candidates':len(candidates),'eligible5':int(eligible.sum()),
            'methods':method_results,'examples':examples,
            'protocol':'Same candidates/users, repeats allowed, ascending-ID ties, shared all60 fallback; empty-target AP is zero only in all-user aggregation',
            'ppr_note':'At most30 iterations with tolerance1e-7; no convergence residual is reported, so treat this as a capped approximation',
            'scope':'One fixed reused temporal comparison, no model selection; no output files written'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT/'config.json')
    parser.add_argument('--user-id',type=int,action='append',default=[],help='Include ordered ID/title/category/target outcomes for this template user; repeat for multiple users')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(compare(args.data_dir.resolve(),config,args.user_id),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

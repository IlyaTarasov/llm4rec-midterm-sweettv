"""Recommendation models for the SWEET.TV midterm.

Every model consumes a TrainData object built only from views before the
evaluation window.  Scores are raw and not comparable across methods; the
evaluator converts each method's scores into a deterministic Top-5 list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class TrainData:
    user_ids: np.ndarray
    item_ids: np.ndarray
    interactions: pd.DataFrame
    item_meta: pd.DataFrame
    n_users: int
    n_items: int
    X_binary: sparse.csr_matrix
    X_count: sparse.csr_matrix
    X_seconds: sparse.csr_matrix
    X_last: sparse.csr_matrix
    window_start_ts: int
    seed: int = 42

    def history_row(self, uidx: int) -> Tuple[np.ndarray, np.ndarray]:
        start, end = self.X_count.indptr[uidx], self.X_count.indptr[uidx + 1]
        return self.X_count.indices[start:end], self.X_count.data[start:end]

    def recency_weights(self, uidx: int, half_life_days: float) -> np.ndarray:
        start, end = self.X_count.indptr[uidx], self.X_count.indptr[uidx + 1]
        idx = self.X_count.indices[start:end]
        if len(idx) == 0:
            return np.zeros(0)
        ts = self.X_last.data[start:end].astype(np.float64)
        ages = np.maximum(0.0, (self.window_start_ts - ts) / 86400.0)
        return np.power(0.5, ages / max(half_life_days, 1e-9))


class BaseModel:
    name = "base"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "BaseModel":
        raise NotImplementedError

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        raise NotImplementedError


class PopularityModel(BaseModel):
    name = "popularity"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "PopularityModel":
        weight = cfg.get("popularity", {}).get("weight", "users")
        if weight == "views":
            self.pop = np.asarray(data.X_count.sum(axis=0)).ravel().astype(float)
        else:
            self.pop = np.asarray(data.X_binary.sum(axis=0)).ravel().astype(float)
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        return np.tile(self.pop[cand_idx], (len(uidx), 1))


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


class AssociationModel(BaseModel):
    name = "association"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "AssociationModel":
        c = cfg.get("association", {})
        self.cfg = cfg
        Xb = data.X_binary
        _, n_items = Xb.shape
        # The shared index includes users whose first view is after this fit
        # cutoff. New models use only users with training interactions. The
        # legacy option exists solely to reproduce/diagnose the first CSV.
        population_mode = c.get("lift_population", "training_active")
        if population_mode == "training_active":
            n_users = int(np.count_nonzero(np.diff(Xb.indptr)))
        elif population_mode == "all_indexed_legacy":
            n_users = int(Xb.shape[0])
        else:
            raise ValueError(f"unknown association lift_population: {population_mode}")
        pop = np.asarray(Xb.sum(axis=0)).ravel().astype(float)
        co = (Xb.T @ Xb).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        min_support = int(c.get("min_support", 5))
        min_conf = float(c.get("min_confidence", 0.05))
        min_lift = float(c.get("min_lift", 1.0))
        top_k = int(c.get("top_k", 100))
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i in range(n_items):
            if pop[i] <= 0:
                continue
            start, end = co.indptr[i], co.indptr[i + 1]
            js = co.indices[start:end]
            cnt = co.data[start:end].astype(float)
            if len(js) == 0:
                continue
            conf = cnt / pop[i]
            lift = cnt * n_users / np.maximum(pop[i] * pop[js], 1e-12)
            keep = (cnt >= min_support) & (conf >= min_conf) & (lift >= min_lift)
            if not keep.any():
                continue
            js, conf = js[keep], conf[keep]
            if len(js) > top_k:
                order = np.lexsort((js, -conf))[:top_k]
                js, conf = js[order], conf[order]
            rows.extend([i] * len(js))
            cols.extend(js.tolist())
            vals.extend(conf.tolist())
        self.S = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        self.pop = pop
        self.n_users = n_users
        self.n_items = n_items
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        half_life = float(self.cfg.get("association", {}).get(
            "half_life_days", self.cfg.get("content", {}).get("half_life_days", 30.0)
        ))
        out = np.zeros((len(uidx), len(cand_idx)), dtype=np.float64)
        for r, u in enumerate(uidx):
            idx, counts = data.history_row(int(u))
            if len(idx) == 0:
                continue
            w = data.recency_weights(int(u), half_life)
            h = sparse.csr_matrix((w, (np.zeros(len(idx), dtype=int), idx)), shape=(1, self.n_items))
            row = np.asarray(h.dot(self.S).todense()).ravel()
            out[r] = row[cand_idx]
        return out


class MFModel(BaseModel):
    name = "mf"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "MFModel":
        c = cfg.get("mf", {})
        k = int(c.get("factors", 64))
        iters = int(c.get("iterations", 15))
        reg = float(c.get("reg", 0.05))
        alpha = float(c.get("alpha", 20.0))
        weight = c.get("weight", "count")
        rng = np.random.default_rng(seed)
        n_users, n_items = data.n_users, data.n_items
        if weight == "binary":
            vals = np.ones(len(data.interactions), dtype=float)
        elif weight == "seconds":
            vals = data.interactions["watched_seconds"].to_numpy(dtype=float) / 60.0
        else:
            vals = np.log1p(data.interactions["view_count"].to_numpy(dtype=float))
        X = sparse.csr_matrix(
            (vals, (data.interactions["uidx"].to_numpy(), data.interactions["iidx"].to_numpy())),
            shape=(n_users, n_items),
        )
        X.sum_duplicates()
        Xc = X.tocsc()
        U = np.zeros((n_users, k), dtype=np.float64)
        V = np.zeros((n_items, k), dtype=np.float64)
        active_u = np.flatnonzero(np.diff(X.indptr) > 0)
        active_i = np.flatnonzero(np.diff(Xc.indptr) > 0)
        U[active_u] = rng.normal(0.0, 0.01, size=(len(active_u), k))
        V[active_i] = rng.normal(0.0, 0.01, size=(len(active_i), k))
        eye = np.eye(k)
        for it in range(iters):
            YtY = V.T @ V
            for u in active_u:
                s, e = X.indptr[u], X.indptr[u + 1]
                idx = X.indices[s:e]
                r = X.data[s:e]
                conf = 1.0 + alpha * r
                Yu = V[idx]
                A = YtY + reg * eye + (Yu * (conf - 1.0)[:, None]).T @ Yu
                b = (Yu * conf[:, None]).sum(axis=0)
                U[u] = np.linalg.solve(A, b)
            XtX = U.T @ U
            for i in active_i:
                s, e = Xc.indptr[i], Xc.indptr[i + 1]
                idx = Xc.indices[s:e]
                r = Xc.data[s:e]
                conf = 1.0 + alpha * r
                Ui = U[idx]
                A = XtX + reg * eye + (Ui * (conf - 1.0)[:, None]).T @ Ui
                b = (Ui * conf[:, None]).sum(axis=0)
                V[i] = np.linalg.solve(A, b)
        self.U, self.V = U, V
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        return self.U[uidx] @ self.V[cand_idx].T


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


class RepeatModel(BaseModel):
    """Recency-weighted repeat prior.

    Score(u, i) = log1p(number of training views of i by u) * 0.5 ** (age_days / half_life).
    It uses only views before the evaluation window.  It is deliberately a
    component, not one of the four required methods: it captures the observed
    re-watching of recurring news programmes.
    """

    name = "repeat"

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "RepeatModel":
        self.cfg = cfg
        self.n_items = data.n_items
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        c = self.cfg.get("repeat", {})
        half_life = float(c.get("half_life_days", self.cfg.get("content", {}).get("half_life_days", 14.0)))
        out = np.zeros((len(uidx), len(cand_idx)), dtype=np.float64)
        if len(cand_idx) == 0:
            return out
        for r, u in enumerate(uidx):
            s, e = data.X_count.indptr[int(u)], data.X_count.indptr[int(u) + 1]
            idx = data.X_count.indices[s:e]
            if len(idx) == 0:
                continue
            counts = data.X_count.data[s:e]
            ts = data.X_last.data[s:e].astype(np.float64)
            w = np.power(0.5, np.maximum(0.0, (data.window_start_ts - ts) / 86400.0) / max(half_life, 1e-9))
            pos = np.searchsorted(cand_idx, idx)
            pos_c = np.clip(pos, 0, len(cand_idx) - 1)
            ok = (pos < len(cand_idx)) & (cand_idx[pos_c] == idx)
            if ok.any():
                out[r, pos[ok]] = np.log1p(counts[ok]) * w[ok]
        return out


class HybridModel(BaseModel):
    name = "hybrid"

    def __init__(self, models: Sequence[BaseModel], weights: Dict[str, float], rrf_k: int = 60):
        self.models = list(models)
        self.weights = dict(weights)
        self.rrf_k = int(rrf_k)

    def fit(self, data: TrainData, cfg: Dict, seed: int) -> "HybridModel":
        for m in self.models:
            if not getattr(m, "fitted", False):
                m.fit(data, cfg, seed)
        self.fitted = True
        return self

    def score(self, uidx: np.ndarray, cand_idx: np.ndarray, data: TrainData) -> np.ndarray:
        out = np.zeros((len(uidx), len(cand_idx)), dtype=np.float64)
        n_cand = len(cand_idx)
        base_ranks = np.arange(1, n_cand + 1, dtype=np.float64)[None, :]
        for m in self.models:
            w = float(self.weights.get(m.name, 0.0))
            if w == 0.0:
                continue
            s = m.score(uidx, cand_idx, data)
            order = np.argsort(-s, axis=1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(base_ranks, order.shape), axis=1)
            out += w / (self.rrf_k + ranks)
        return out

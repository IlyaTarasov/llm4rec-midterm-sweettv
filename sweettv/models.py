"""Popularity, association rules and implicit ALS matrix factorization."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from dataclasses import dataclass

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

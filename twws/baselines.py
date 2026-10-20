"""End-to-end baselines.

All baselines share the protocol of the proposed detector:

* the same partition and the same pseudo-labels (ground truth is used for evaluation only);
* trust scores enter as sample weights wherever the loss admits weighting;
* the full multi-view feature matrix (before feature selection) as input, so that each model
  can learn its own representation; sequence models read the ``seq_window`` most recent flows
  of the same source host, the graph model reads the host graph;
* the same search routine and number of trials (``twws.tuning.run_search``);
* early stopping / model selection on pseudo-labelled validation rows.

``BASELINES`` maps a key to (display name, factory).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .metrics import best_f1_threshold
from .model import train_twxgb
from .utils import get_logger

DL_ES_MAX_ROWS = 50_000      # validation flows used for early stopping of the neural models, TabNet and E-GraphSAGE
GRAPH_SCORE_CHUNK = 400_000  # flows scored together (one graph per chunk) by the E-GraphSAGE baseline

log = get_logger("twws.baselines")


def _clean(X: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -30.0, 30.0).astype(np.float32)


def _balanced_weights(y: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """Trust weight times a class weight that gives both pseudo-classes equal total weight."""
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
    n1, n0 = max(1, y.sum()), max(1, len(y) - y.sum())
    cw = np.where(y == 1, len(y) / (2.0 * n1), len(y) / (2.0 * n0))
    return (w * cw).astype(np.float32)


class Baseline:
    key = "base"
    name = "Baseline"
    space: dict = {}
    uses_sequence = False
    uses_graph = False

    def __init__(self, params: dict | None = None, seed: int = 0, cfg=None, n_jobs: int = -1) -> None:
        self.params = dict(params or {})
        self.seed, self.cfg, self.n_jobs = seed, cfg, n_jobs
        self.fit_seconds = 0.0
        self.n_params: int | str | None = None

    def fit(self, prep, tr: np.ndarray, es: np.ndarray, budget: float = 1.0) -> "Baseline":
        raise NotImplementedError

    def score(self, prep, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"model": self.name, "params": self.params, "n_parameters": self.n_params,
                "fit_seconds": round(self.fit_seconds, 1)}


# --------------------------------------------------------------------------- tree ensembles
class RFBaseline(Baseline):
    key, name = "rf", "Random Forest"
    space = {"n_estimators": {"type": "categorical", "choices": [200, 400]},
             "max_depth": {"type": "int", "low": 8, "high": 24},
             "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
             "max_features": {"type": "categorical", "choices": ["sqrt", 0.3, 0.5]}}

    def fit(self, prep, tr, es, budget=1.0):
        from sklearn.ensemble import RandomForestClassifier

        t0 = time.perf_counter()
        p = {"n_estimators": 400, "max_depth": 24, "min_samples_leaf": 1, "max_features": "sqrt", **self.params}
        n_est = max(50, int(p["n_estimators"] * budget))
        self.m = RandomForestClassifier(n_estimators=n_est, max_depth=int(p["max_depth"]),
                                        min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
                                        n_jobs=self.n_jobs, random_state=self.seed)
        self.m.fit(_clean(prep.X[tr]), prep.y_pl[tr], sample_weight=_balanced_weights(prep.y_pl[tr], prep.w_pl[tr]))
        self.n_params = f"{n_est} trees"
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows, chunk=300_000):
        out = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), chunk):
            out[i: i + chunk] = self.m.predict_proba(_clean(prep.X[rows[i: i + chunk]]))[:, 1]
        return out


class LGBMBaseline(Baseline):
    key, name = "lightgbm", "LightGBM"
    space = {"num_leaves": {"type": "int", "low": 15, "high": 255},
             "learning_rate": {"type": "loguniform", "low": 0.02, "high": 0.3},
             "min_child_samples": {"type": "int", "low": 10, "high": 200},
             "feature_fraction": {"type": "uniform", "low": 0.5, "high": 1.0},
             "bagging_fraction": {"type": "uniform", "low": 0.5, "high": 1.0},
             "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 10.0}}

    def fit(self, prep, tr, es, budget=1.0):
        import lightgbm as lgb

        t0 = time.perf_counter()
        p = {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "reg_lambda": 1.0, **self.params}
        params = {"objective": "binary", "metric": "average_precision", "verbose": -1, "seed": self.seed,
                  "num_threads": self.n_jobs if self.n_jobs and self.n_jobs > 0 else 0, "bagging_freq": 1,
                  "num_leaves": int(p["num_leaves"]), "learning_rate": float(p["learning_rate"]),
                  "min_child_samples": int(p["min_child_samples"]), "feature_fraction": float(p["feature_fraction"]),
                  "bagging_fraction": float(p["bagging_fraction"]), "lambda_l2": float(p["reg_lambda"])}
        dtr = lgb.Dataset(prep.X[tr], prep.y_pl[tr], weight=_balanced_weights(prep.y_pl[tr], prep.w_pl[tr]))
        des = lgb.Dataset(prep.X[es], prep.y_pl[es], reference=dtr)
        rounds = max(50, int(1000 * budget))
        self.m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[des],
                           callbacks=[lgb.early_stopping(50 if budget >= 1 else 20, verbose=False)])
        self.n_params = f"{self.m.best_iteration or rounds} trees"
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows):
        return self.m.predict(prep.X[rows], num_iteration=self.m.best_iteration).astype(np.float32)


class XGBLoglossBaseline(Baseline):
    """XGBoost with the plain logistic loss; trust and class balance enter as sample weights."""
    key, name = "xgb_logloss", "XGBoost (logistic loss)"
    space = {"eta": {"type": "loguniform", "low": 0.02, "high": 0.3},
             "max_depth": {"type": "int", "low": 3, "high": 10},
             "min_child_weight": {"type": "loguniform", "low": 0.1, "high": 10.0},
             "gamma_split": {"type": "uniform", "low": 0.0, "high": 5.0},
             "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 10.0},
             "colsample_bytree": {"type": "uniform", "low": 0.5, "high": 1.0},
             "subsample": {"type": "uniform", "low": 0.5, "high": 1.0}}

    def fit(self, prep, tr, es, budget=1.0):
        p = {**self.params, "focal_gamma": 0.0, "delta": None, "class_weight": "balanced", "use_trust": True,
             "warmup_rounds": 0, "num_boost_round": max(50, int(800 * budget)),
             "early_stopping_rounds": 50 if budget >= 1 else 20}
        self.m = train_twxgb(prep.X[tr], prep.y_pl[tr], prep.w_pl[tr], prep.X[es], prep.y_pl[es], p, seed=self.seed,
                             nthread=self.n_jobs if self.n_jobs and self.n_jobs > 0 else None)
        self.n_params = f"{self.m.n_trees} trees"
        self.fit_seconds = self.m.fit_seconds
        return self

    def score(self, prep, rows):
        return self.m.predict_proba(prep.X[rows])


# --------------------------------------------------------------------------- neural baselines (PyTorch)
def host_windows(prep, meta, window: int) -> np.ndarray:
    """For every corpus row, the indices of the ``window`` most recent corpus rows of the same
    source host up to and including the row itself (-1 = padding at the front)."""
    cached = getattr(prep, "_seq_idx", None)
    if cached is not None and cached.shape[1] == window:
        return cached
    import pandas as pd

    host = pd.factorize(meta["src_ip"].astype(str))[0]
    t = meta["t_avail"].to_numpy(dtype=np.float64)
    order = np.lexsort((np.arange(len(t)), t, host))
    hs = host[order]
    start = np.zeros(len(order), dtype=np.int64)
    cut = np.flatnonzero(np.diff(hs)) + 1
    seg_start = np.concatenate(([0], cut))
    seg_id = np.searchsorted(seg_start, np.arange(len(order)), side="right") - 1
    start = seg_start[seg_id]
    pos = np.arange(len(order))
    idx = np.full((len(order), window), -1, dtype=np.int64)
    for k in range(window):
        src = pos - (window - 1 - k)
        ok = src >= start
        idx[ok, k] = order[src[ok]]
    out = np.empty_like(idx)
    out[order] = idx
    prep._seq_idx = out
    return out


class TorchBaseline(Baseline):
    """Shared training loop: weighted BCE, Adam, early stopping on pseudo-label PR-AUC."""
    space = {"lr": {"type": "loguniform", "low": 3e-4, "high": 3e-3},
             "hidden": {"type": "categorical", "choices": [64, 128]},
             "dropout": {"type": "uniform", "low": 0.0, "high": 0.3}}
    batch_size = 1024

    def build(self, d_in: int):
        raise NotImplementedError

    def _batch(self, prep, rows):
        import torch

        return torch.from_numpy(_clean(prep.X[rows]))

    def _prepare(self, prep):
        pass

    def fit(self, prep, tr, es, budget=1.0):
        import torch
        from sklearn.metrics import average_precision_score

        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        if self.n_jobs and self.n_jobs > 0:
            torch.set_num_threads(self.n_jobs)
        rng = np.random.default_rng(self.seed)
        cap = int(self.cfg.experiments.dl_max_train_rows) if self.cfg is not None else 300_000
        if len(tr) > cap:
            tr = np.sort(rng.choice(tr, size=cap, replace=False))
        if len(es) > DL_ES_MAX_ROWS:
            es = np.sort(rng.choice(es, size=DL_ES_MAX_ROWS, replace=False))
        self._prepare(prep)
        self.net = self.build(prep.X.shape[1])
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.params.get("lr", 1e-3)))
        w_all = _balanced_weights(prep.y_pl[tr], prep.w_pl[tr])
        epochs = max(1, int(round((int(self.cfg.experiments.dl_epochs) if self.cfg is not None else 8) * budget)))
        best, best_state, bad = -1.0, None, 0
        y_es = prep.y_pl[es].astype(int)
        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(len(tr))
            for i in range(0, len(perm), self.batch_size):
                b = perm[i: i + self.batch_size]
                xb = self._batch(prep, tr[b])
                yb = torch.from_numpy(prep.y_pl[tr[b]].astype(np.float32))
                wb = torch.from_numpy(w_all[b])
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(xb).reshape(-1), yb, weight=wb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            s = self.score(prep, es)
            ap = float(average_precision_score(y_es, s)) if 0 < y_es.sum() < len(y_es) else 0.0
            if ap > best + 1e-5:
                best, bad = ap, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.epochs_run = ep + 1
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows, chunk=8192):
        import torch

        self.net.eval()
        out = np.empty(len(rows), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), chunk):
                out[i: i + chunk] = torch.sigmoid(self.net(self._batch(prep, rows[i: i + chunk])).reshape(-1)).numpy()
        return out

    def describe(self):
        d = super().describe()
        d["epochs_run"] = getattr(self, "epochs_run", None)
        return d


class SeqBaseline(TorchBaseline):
    uses_sequence = True

    def _prepare(self, prep):
        self.window = int(self.cfg.experiments.seq_window) if self.cfg is not None else 16
        self.seq_idx = host_windows(prep, prep.meta_ref, self.window)
        if getattr(prep, "_Xpad", None) is None:
            prep._Xpad = np.vstack([_clean(prep.X), np.zeros((1, prep.X.shape[1]), dtype=np.float32)])
        self.Xpad = prep._Xpad

    def _batch(self, prep, rows):
        import torch

        idx = self.seq_idx[rows]
        idx = np.where(idx < 0, len(self.Xpad) - 1, idx)
        return torch.from_numpy(self.Xpad[idx])                 # (B, W, d)


class CNN1DBaseline(SeqBaseline):
    key, name = "cnn1d", "1D-CNN"

    def build(self, d_in):
        import torch.nn as nn

        h = int(self.params.get("hidden", 128))
        dr = float(self.params.get("dropout", 0.1))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv1d(d_in, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, h, 3, padding=1), nn.ReLU(),
                                    nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
                s.f = nn.Sequential(nn.Dropout(dr), nn.Linear(2 * h, 1))

            def forward(s, x):
                z = s.c(x.transpose(1, 2))                       # (B, h, W)
                return s.f(__import__("torch").cat([z.max(dim=2).values, z[:, :, -1]], dim=1))
        return Net()


class LSTMBaseline(SeqBaseline):
    key, name = "lstm", "LSTM"

    def build(self, d_in):
        import torch.nn as nn

        h = int(self.params.get("hidden", 128))
        dr = float(self.params.get("dropout", 0.1))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.l = nn.LSTM(d_in, h, num_layers=2, batch_first=True, dropout=dr)
                s.f = nn.Linear(h, 1)

            def forward(s, x):
                o, _ = s.l(x)
                return s.f(o[:, -1])
        return Net()


class CNNLSTMBaseline(SeqBaseline):
    key, name = "cnn_lstm", "CNN-LSTM"

    def build(self, d_in):
        import torch.nn as nn

        h = int(self.params.get("hidden", 128))
        dr = float(self.params.get("dropout", 0.1))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv1d(d_in, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, h, 3, padding=1), nn.ReLU())
                s.l = nn.LSTM(h, h, batch_first=True)
                s.f = nn.Sequential(nn.Dropout(dr), nn.Linear(h, 1))

            def forward(s, x):
                z = s.c(x.transpose(1, 2)).transpose(1, 2)
                o, _ = s.l(z)
                return s.f(o[:, -1])
        return Net()


class TabNetBaseline(Baseline):
    key, name = "tabnet", "TabNet"
    space = {"n_steps": {"type": "int", "low": 3, "high": 6},
             "width": {"type": "categorical", "choices": [16, 32, 64]},
             "lr": {"type": "loguniform", "low": 5e-3, "high": 3e-2},
             "lambda_sparse": {"type": "loguniform", "low": 1e-4, "high": 1e-2}}

    def fit(self, prep, tr, es, budget=1.0):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        t0 = time.perf_counter()
        if self.n_jobs and self.n_jobs > 0:
            torch.set_num_threads(self.n_jobs)
        rng = np.random.default_rng(self.seed)
        cap = int(self.cfg.experiments.dl_max_train_rows) if self.cfg is not None else 300_000
        if len(tr) > cap:
            tr = np.sort(rng.choice(tr, size=cap, replace=False))
        if len(es) > DL_ES_MAX_ROWS:
            es = np.sort(rng.choice(es, size=DL_ES_MAX_ROWS, replace=False))
        p = {"n_steps": 5, "width": 64, "lr": 2e-2, "lambda_sparse": 1e-3, **self.params}
        self.m = TabNetClassifier(n_d=int(p["width"]), n_a=int(p["width"]), n_steps=int(p["n_steps"]),
                                  lambda_sparse=float(p["lambda_sparse"]), optimizer_params={"lr": float(p["lr"])},
                                  seed=self.seed, verbose=0)
        epochs = max(1, int(round((int(self.cfg.experiments.dl_epochs) if self.cfg is not None else 8) * budget)))
        w = _balanced_weights(prep.y_pl[tr], prep.w_pl[tr]).astype(np.float64)
        from pytorch_tabnet.metrics import Metric
        from sklearn.metrics import average_precision_score

        class PseudoLabelPRAUC(Metric):          # the early-stopping criterion of every other model
            def __init__(self):
                self._name = "pr_auc"
                self._maximize = True

            def __call__(self, y_true, y_score):
                yt = np.asarray(y_true).astype(int)
                return float(average_precision_score(yt, y_score[:, 1])) if 0 < yt.sum() < len(yt) else 0.0

        # pytorch-tabnet turns per-row weights into a weighted sampler of training rows (not into loss weights)
        self.m.fit(_clean(prep.X[tr]), prep.y_pl[tr].astype(int), eval_set=[(_clean(prep.X[es]), prep.y_pl[es].astype(int))],
                   eval_metric=[PseudoLabelPRAUC], max_epochs=epochs, patience=2, batch_size=4096, virtual_batch_size=256,
                   weights=w / w.sum() * len(w))
        self.n_params = int(sum(q.numel() for q in self.m.network.parameters()))
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows, chunk=200_000):
        out = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), chunk):
            out[i: i + chunk] = self.m.predict_proba(_clean(prep.X[rows[i: i + chunk]]))[:, 1]
        return out


class EGraphSAGEBaseline(Baseline):
    """Edge-classification GraphSAGE on the host graph (hosts = nodes, flows = edges).

    Node states start as ones; each of the two layers averages, for every host, the
    concatenation [neighbour state || flow features] over its incident flows and applies a
    linear map with ReLU; a flow is scored from [state(src) || state(dst) || flow features].
    Message passing uses only the flows of the set being processed (training flows when
    fitting, the scored flows at evaluation), so no label or feature crosses subsets.
    """
    key, name = "egraphsage", "E-GraphSAGE"
    uses_graph = True
    space = {"lr": {"type": "loguniform", "low": 1e-3, "high": 1e-2},
             "hidden": {"type": "categorical", "choices": [64, 128]},
             "dropout": {"type": "uniform", "low": 0.0, "high": 0.3}}

    def _graph(self, prep, rows):
        import pandas as pd
        import torch

        meta = prep.meta_ref
        both = pd.concat([meta["src_ip"].iloc[rows].astype(str), meta["dst_ip"].iloc[rows].astype(str)], ignore_index=True)
        codes, uniq = pd.factorize(both)
        n = len(rows)
        return (torch.from_numpy(codes[:n].astype(np.int64)), torch.from_numpy(codes[n:].astype(np.int64)), len(uniq),
                torch.from_numpy(_clean(prep.X[rows])))

    def _build(self, d_in):
        import torch
        import torch.nn as nn

        h = int(self.params.get("hidden", 128))
        dr = float(self.params.get("dropout", 0.1))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.h = h
                s.l1 = nn.Linear(h + (h + d_in), h)
                s.l2 = nn.Linear(h + (h + d_in), h)
                s.out = nn.Sequential(nn.Dropout(dr), nn.Linear(2 * h + d_in, h), nn.ReLU(), nn.Linear(h, 1))

            def layer(s, lin, state, src, dst, e, n_nodes):
                msg_to_dst = torch.cat([state[src], e], dim=1)
                msg_to_src = torch.cat([state[dst], e], dim=1)
                agg = torch.zeros(n_nodes, msg_to_dst.shape[1])
                deg = torch.zeros(n_nodes)
                agg.index_add_(0, dst, msg_to_dst)
                agg.index_add_(0, src, msg_to_src)
                deg.index_add_(0, dst, torch.ones(len(dst)))
                deg.index_add_(0, src, torch.ones(len(src)))
                agg = agg / deg.clamp(min=1).unsqueeze(1)
                return torch.relu(lin(torch.cat([state, agg], dim=1)))

            def forward(s, src, dst, n_nodes, e):
                state = torch.ones(n_nodes, s.h)
                state = s.layer(s.l1, state, src, dst, e, n_nodes)
                state = s.layer(s.l2, state, src, dst, e, n_nodes)
                return s.out(torch.cat([state[src], state[dst], e], dim=1)).reshape(-1)
        return Net()

    def fit(self, prep, tr, es, budget=1.0):
        import torch
        from sklearn.metrics import average_precision_score

        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        if self.n_jobs and self.n_jobs > 0:
            torch.set_num_threads(self.n_jobs)
        rng = np.random.default_rng(self.seed)
        cap = int(self.cfg.experiments.dl_max_train_rows) if self.cfg is not None else 300_000
        if len(tr) > cap:
            tr = np.sort(rng.choice(tr, size=cap, replace=False))
        if len(es) > DL_ES_MAX_ROWS:
            es = np.sort(rng.choice(es, size=DL_ES_MAX_ROWS, replace=False))
        src, dst, n_nodes, e = self._graph(prep, tr)
        y = torch.from_numpy(prep.y_pl[tr].astype(np.float32))
        w = torch.from_numpy(_balanced_weights(prep.y_pl[tr], prep.w_pl[tr]))
        self.net = self._build(prep.X.shape[1])
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.params.get("lr", 3e-3)))
        steps = max(10, int(round(20 * (int(self.cfg.experiments.dl_epochs) if self.cfg is not None else 8) * budget)))
        best, best_state, bad = -1.0, None, 0
        y_es = prep.y_pl[es].astype(int)
        for step in range(steps):
            self.net.train()
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(src, dst, n_nodes, e), y, weight=w)
            loss.backward()
            opt.step()
            if (step + 1) % 10 == 0:
                s = self.score(prep, es)
                ap = float(average_precision_score(y_es, s)) if 0 < y_es.sum() < len(y_es) else 0.0
                if ap > best + 1e-5:
                    best, bad = ap, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 3:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows, chunk=GRAPH_SCORE_CHUNK):
        import torch

        self.net.eval()
        out = np.empty(len(rows), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), chunk):
                r = rows[i: i + chunk]
                src, dst, n_nodes, e = self._graph(prep, r)
                out[i: i + chunk] = torch.sigmoid(self.net(src, dst, n_nodes, e)).numpy()
        return out


class KitNETBaseline(Baseline):
    """KitNET, the detector core of Kitsune, applied to the flow feature matrix.

    Features are grouped by hierarchical clustering on correlation distance (at most
    ``max_group`` features per group); one small autoencoder per group and an output
    autoencoder over the per-group reconstruction errors give the anomaly score (RMSE).
    It is trained on training flows pseudo-labelled benign; no label enters the loss.
    The decision threshold is calibrated like every other model.
    """
    key, name = "kitnet", "Kitsune (KitNET)"
    space = {"max_group": {"type": "categorical", "choices": [5, 10]},
             "hidden_ratio": {"type": "categorical", "choices": [0.5, 0.75]},
             "lr": {"type": "loguniform", "low": 1e-3, "high": 1e-2}}

    def _groups(self, X, m):
        from scipy.cluster.hierarchy import fcluster, linkage

        sd = X.std(axis=0)
        ok = np.flatnonzero(sd > 0)
        if len(ok) < 2:
            return [list(range(X.shape[1]))]
        C = np.nan_to_num(np.corrcoef(X[:, ok], rowvar=False), nan=0.0)
        D = 1.0 - np.abs(C)
        iu = np.triu_indices(len(ok), 1)
        Z = linkage(np.clip(D[iu], 0, None), method="average")
        for k in range(max(1, math.ceil(len(ok) / m)), len(ok) + 1):
            lab = fcluster(Z, t=k, criterion="maxclust")
            if np.bincount(lab).max() <= m:
                break
        groups = [ok[lab == c].tolist() for c in np.unique(lab)]
        const = [i for i in range(X.shape[1]) if i not in set(ok.tolist())]
        if const:
            groups.append(const)
        return groups

    def fit(self, prep, tr, es, budget=1.0):
        import torch
        import torch.nn as nn

        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        if self.n_jobs and self.n_jobs > 0:
            torch.set_num_threads(self.n_jobs)
        rng = np.random.default_rng(self.seed)
        ben = tr[prep.y_pl[tr] == 0]
        cap = int(self.cfg.experiments.dl_max_train_rows) if self.cfg is not None else 300_000
        if len(ben) > cap:
            ben = np.sort(rng.choice(ben, size=cap, replace=False))
        X = np.clip(_clean(prep.X[ben]), -5, 5)
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        self.rng_ = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        Xn = (X - self.lo) / self.rng_
        m = int(self.params.get("max_group", 10))
        ratio = float(self.params.get("hidden_ratio", 0.75))
        self.groups = self._groups(Xn[: min(len(Xn), 50_000)], m)

        class Net(nn.Module):
            def __init__(s, groups):
                super().__init__()
                s.groups = [torch.tensor(g, dtype=torch.long) for g in groups]
                s.enc = nn.ModuleList([nn.Linear(len(g), max(1, math.ceil(ratio * len(g)))) for g in groups])
                s.dec = nn.ModuleList([nn.Linear(max(1, math.ceil(ratio * len(g))), len(g)) for g in groups])
                G = len(groups)
                s.oe = nn.Linear(G, max(1, math.ceil(ratio * G)))
                s.od = nn.Linear(max(1, math.ceil(ratio * G)), G)

            def forward(s, x):
                errs, rec_loss = [], 0.0
                for g, e, d in zip(s.groups, s.enc, s.dec):
                    xg = x[:, g]
                    r = torch.sigmoid(d(torch.sigmoid(e(xg))))
                    se = ((r - xg) ** 2).mean(dim=1)
                    errs.append(torch.sqrt(se + 1e-12))
                    rec_loss = rec_loss + se.mean()
                E = torch.stack(errs, dim=1)
                Ed = E.detach()
                ro = torch.sigmoid(s.od(torch.sigmoid(s.oe(Ed))))
                out_se = ((ro - Ed) ** 2).mean(dim=1)
                return torch.sqrt(out_se + 1e-12), rec_loss + out_se.mean()
        self.net = Net(self.groups)
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.params.get("lr", 3e-3)))
        Xt = torch.from_numpy(Xn.astype(np.float32))
        epochs = max(1, int(round((int(self.cfg.experiments.dl_epochs) if self.cfg is not None else 8) * budget)))
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 1024):
                opt.zero_grad()
                _, loss = self.net(Xt[perm[i: i + 1024]])
                loss.backward()
                opt.step()
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, prep, rows, chunk=100_000):
        import torch

        self.net.eval()
        out = np.empty(len(rows), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), chunk):
                X = (np.clip(_clean(prep.X[rows[i: i + chunk]]), -5, 5) - self.lo) / self.rng_
                out[i: i + chunk] = self.net(torch.from_numpy(X.astype(np.float32)))[0].numpy()
        return out

    def describe(self):
        d = super().describe()
        d["n_groups"] = len(getattr(self, "groups", []))
        return d


BASELINES = {c.key: c for c in [RFBaseline, LGBMBaseline, XGBLoglossBaseline, CNN1DBaseline, LSTMBaseline,
                                CNNLSTMBaseline, TabNetBaseline, EGraphSAGEBaseline, KitNETBaseline]}


def fit_baseline(key: str, prep, params: dict, seed: int, cfg, n_jobs: int, budget: float = 1.0,
                 tr: np.ndarray | None = None, es: np.ndarray | None = None):
    cls = BASELINES[key]
    b = cls(params, seed=seed, cfg=cfg, n_jobs=n_jobs)
    tr = np.flatnonzero(prep.usable_train()) if tr is None else tr
    es = np.flatnonzero(prep.val_a) if es is None else es
    return b.fit(prep, tr, es, budget=budget)


def baseline_threshold(b: Baseline, prep) -> tuple[float, float]:
    rows = np.flatnonzero(prep.val_b)
    return best_f1_threshold(prep.y_pl[rows], b.score(prep, rows))

"""Step 5: stability-aware feature selection, and the selection baselines it is compared with.

Ground-truth labels are never used here; relevance is measured against pseudo-labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .utils import get_logger

log = get_logger("twws.fselect")


# --------------------------------------------------------------------------- relevance measures
def binned_mutual_information(X: np.ndarray, y: np.ndarray, bins: int = 32) -> np.ndarray:
    """Mutual information (nats) between each feature, discretised into quantile bins, and y.

    This is Eq. (14) evaluated on the empirical joint distribution of (bin, pseudo-label);
    missing values form their own bin.
    """
    y = np.asarray(y).astype(np.int64)
    n, d = X.shape
    py = np.bincount(y, minlength=2) / n
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    out = np.zeros(d)
    for j in range(d):
        x = X[:, j].astype(np.float64)
        nan = np.isnan(x)
        if nan.all():
            continue
        edges = np.unique(np.quantile(x[~nan], qs))
        code = np.searchsorted(edges, x, side="right")
        code[nan] = len(edges) + 1
        nb = len(edges) + 2
        joint = np.bincount(code * 2 + y, minlength=nb * 2).reshape(nb, 2) / n
        pb = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * np.log(joint / (pb * py[None, :]))
        out[j] = np.nansum(term)
    return out


def rf_importance(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_estimators: int = 100,
                  max_depth: int = 16, max_samples: int | None = 200_000, seed: int = 0, n_jobs: int = -1) -> np.ndarray:
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ms = None if (max_samples is None or max_samples >= len(Xc)) else int(max_samples)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_samples=ms, n_jobs=n_jobs,
                                random_state=seed, class_weight="balanced_subsample", min_samples_leaf=5)
    rf.fit(Xc, y, sample_weight=w)
    return rf.feature_importances_


# --------------------------------------------------------------------------- proposed selector
@dataclass
class StabilitySelector:
    """MI filter -> random-forest ranking -> cross-fold stability -> borderline re-evaluation."""
    mi_top: int = 80
    rf_top: int = 40
    n_folds: int = 5
    theta: float = 0.8
    borderline_low: float = 0.5
    rf_trees: int = 100
    rf_depth: int = 16
    rf_max_samples: int | None = 200_000
    max_rows: int | None = 600_000
    bins: int = 32
    use_stability: bool = True
    use_borderline: bool = True
    seed: int = 0
    n_jobs: int = -1
    info: dict = field(default_factory=dict)

    def _one(self, X, y, w, seed):
        mi = binned_mutual_information(X, y, self.bins)
        keep = np.argsort(-mi, kind="stable")[: min(self.mi_top, X.shape[1])]
        imp = rf_importance(X[:, keep], y, w, self.rf_trees, self.rf_depth, self.rf_max_samples, seed, self.n_jobs)
        top = keep[np.argsort(-imp, kind="stable")[: min(self.rf_top, len(keep))]]
        return mi, keep, imp, top

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, names: list[str],
            high_trust: np.ndarray | None = None) -> "StabilitySelector":
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if self.max_rows and len(idx) > self.max_rows:
            idx = np.sort(rng.choice(idx, size=self.max_rows, replace=False))
        Xs, ys = X[idx], np.asarray(y)[idx].astype(int)
        ws = None if w is None else np.asarray(w)[idx]
        d = X.shape[1]
        mi_all, _, _, top_all = self._one(Xs, ys, ws, self.seed)
        if not self.use_stability:
            sel = np.sort(top_all)
            self.selected_ = sel
            self.scores_ = np.isin(np.arange(d), sel).astype(float)
            self.info = {"mode": "single-pass (no stability)", "n_selected": int(len(sel))}
            self.names_ = [names[i] for i in sel]
            return self
        score = np.zeros(d)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_sets = []
        for f, (tr, _) in enumerate(skf.split(Xs, ys)):
            _, _, _, top = self._one(Xs[tr], ys[tr], None if ws is None else ws[tr], self.seed + 1 + f)
            score[top] += 1
            fold_sets.append(sorted(int(i) for i in top))
        score /= self.n_folds
        stable = np.flatnonzero(score >= self.theta - 1e-9)
        border = np.flatnonzero((score >= self.borderline_low - 1e-9) & (score < self.theta - 1e-9))
        kept_border = np.array([], dtype=int)
        if self.use_borderline and len(border) and len(stable):
            cand = np.concatenate([stable, border])
            mask = np.ones(len(ys), bool) if high_trust is None else np.asarray(high_trust)[idx]
            Xl = np.clip(np.nan_to_num(Xs[mask][:, cand], nan=0.0, posinf=0.0, neginf=0.0), -30, 30)
            yl = ys[mask]
            if len(np.unique(yl)) == 2:
                sd = Xl.std(axis=0)
                sd[sd == 0] = 1.0
                lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xl - Xl.mean(axis=0)) / sd, yl)
                coef = np.abs(lr.coef_[0])
                ref = np.median(coef[: len(stable)])
                kept_border = border[coef[len(stable):] >= ref]
                self.info["borderline_rule"] = {"reference_abs_coef": float(ref),
                                                "border_abs_coef": coef[len(stable):].tolist()}
        sel = np.sort(np.concatenate([stable, kept_border])).astype(int)
        if len(sel) == 0:                      # degenerate case: fall back to the single-pass top-k
            sel = np.sort(top_all)
        self.selected_ = sel
        self.scores_ = score
        self.names_ = [names[i] for i in sel]
        self.info.update({
            "mode": "stability", "mi_top": self.mi_top, "rf_top": self.rf_top, "n_folds": self.n_folds,
            "theta": self.theta, "borderline_low": self.borderline_low, "rows_used": int(len(idx)),
            "n_stable": int(len(stable)), "n_borderline": int(len(border)), "n_borderline_kept": int(len(kept_border)),
            "n_selected": int(len(sel)),
            "stability_scores": {names[i]: float(score[i]) for i in np.flatnonzero(score > 0)},
            "mutual_information": {names[i]: float(mi_all[i]) for i in range(d)},
        })
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_]


# --------------------------------------------------------------------------- baselines
class IdentitySelector:
    def __init__(self):
        self.info = {"mode": "all features"}

    def fit(self, X, y, w, names, **kw):
        self.names_ = list(names)
        self.selected_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        return X


class TopKSelector:
    """Single relevance ranking without stability analysis ('mi' or 'rf')."""

    def __init__(self, k: int = 40, kind: str = "mi", seed: int = 0, max_rows: int | None = 600_000, n_jobs: int = -1):
        self.k, self.kind, self.seed, self.max_rows, self.n_jobs = k, kind, seed, max_rows, n_jobs
        self.info = {"mode": f"top-{k} by {kind}", "max_rows": max_rows}

    def fit(self, X, y, w, names, **kw):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        if self.kind == "mi":
            s = binned_mutual_information(X[idx], np.asarray(y)[idx])
        else:
            s = rf_importance(X[idx], np.asarray(y)[idx], None if w is None else np.asarray(w)[idx], seed=self.seed,
                              n_jobs=self.n_jobs)
        self.selected_ = np.sort(np.argsort(-s, kind="stable")[: self.k])
        self.names_ = [names[i] for i in self.selected_]
        return self

    def transform(self, X):
        return X[:, self.selected_]


class PCASelector:
    def __init__(self, k: int = 40, seed: int = 0, max_rows: int | None = 600_000):
        self.k, self.seed, self.max_rows = k, seed, max_rows
        self.info = {"mode": f"PCA({k})", "max_rows": max_rows}

    def fit(self, X, y, w, names, **kw):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(X))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        Xc = np.clip(np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0), -30, 30)
        self.pca_ = PCA(n_components=min(self.k, X.shape[1]), random_state=self.seed).fit(Xc)
        self.names_ = [f"pc_{i:02d}" for i in range(self.pca_.n_components_)]
        self.info["explained_variance"] = float(self.pca_.explained_variance_ratio_.sum())
        return self

    def transform(self, X, chunk: int = 500_000):
        out = np.empty((len(X), self.pca_.n_components_), dtype=np.float32)
        for i in range(0, len(X), chunk):
            out[i: i + chunk] = self.pca_.transform(np.clip(np.nan_to_num(X[i: i + chunk], nan=0.0, posinf=0.0, neginf=0.0), -30, 30))
        return out


class RFESelector:
    """Recursive feature elimination with a random forest, 10% of the features per step."""

    def __init__(self, k: int = 40, seed: int = 0, max_rows: int | None = 150_000, n_jobs: int = -1):
        self.k, self.seed, self.max_rows, self.n_jobs = k, seed, max_rows, n_jobs
        self.info = {"mode": f"RFE to {k} (random forest, 60 trees, depth 12, step 10%)", "max_rows": max_rows}

    def fit(self, X, y, w, names, **kw):
        from sklearn.feature_selection import RFE

        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        est = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=self.n_jobs, random_state=self.seed,
                                     class_weight="balanced_subsample", min_samples_leaf=5)
        rfe = RFE(est, n_features_to_select=min(self.k, X.shape[1]), step=0.1)
        rfe.fit(np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0), np.asarray(y)[idx])
        self.selected_ = np.flatnonzero(rfe.support_)
        self.names_ = [names[i] for i in self.selected_]
        return self

    def transform(self, X):
        return X[:, self.selected_]


class BorutaSelector:
    """Boruta all-relevant selection; confirmed features, completed/truncated to k by ranking."""

    def __init__(self, k: int = 40, seed: int = 0, max_rows: int | None = 100_000, max_iter: int = 30, n_jobs: int = -1):
        self.k, self.seed, self.max_rows, self.max_iter, self.n_jobs = k, seed, max_rows, max_iter, n_jobs
        self.info = {"mode": f"Boruta (100 trees, depth 7, max_iter={max_iter}), top-{k}", "max_rows": max_rows}

    def fit(self, X, y, w, names, **kw):
        from boruta import BorutaPy

        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        est = RandomForestClassifier(n_estimators=100, max_depth=7, n_jobs=self.n_jobs, random_state=self.seed,
                                     class_weight="balanced_subsample")
        b = BorutaPy(est, n_estimators=100, max_iter=self.max_iter, random_state=self.seed, verbose=0)
        Xc = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0)
        b.fit(Xc, np.asarray(y)[idx])
        est.fit(Xc, np.asarray(y)[idx])
        order = np.lexsort((-est.feature_importances_, b.ranking_))      # ranking first, importance as tie-break
        self.selected_ = np.sort(order[: min(self.k, X.shape[1])])
        self.names_ = [names[i] for i in self.selected_]
        self.info.update({"confirmed": int(b.support_.sum()), "tentative": int(b.support_weak_.sum())})
        return self

    def transform(self, X):
        return X[:, self.selected_]


class AESelector:
    """Autoencoder-based ranking: features with the largest first-layer encoder weight norm."""

    def __init__(self, k: int = 40, seed: int = 0, max_rows: int | None = 300_000, epochs: int = 15):
        self.k, self.seed, self.max_rows, self.epochs = k, seed, max_rows, epochs
        self.info = {"mode": f"autoencoder weight-norm ranking, top-{k}", "max_rows": max_rows, "epochs": epochs}

    def fit(self, X, y, w, names, **kw):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(X))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        Xc = np.clip(np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0), -20, 20).astype(np.float32)
        sd = Xc.std(axis=0)
        sd[sd == 0] = 1.0
        Xt = torch.from_numpy(Xc / sd)
        d = X.shape[1]
        enc1 = torch.nn.Linear(d, 64)
        net = torch.nn.Sequential(enc1, torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, d))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 2048):
                xb = Xt[perm[i: i + 2048]]
                opt.zero_grad()
                loss = ((net(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
        score = enc1.weight.detach().norm(dim=0).numpy()
        self.selected_ = np.sort(np.argsort(-score, kind="stable")[: min(self.k, d)])
        self.names_ = [names[i] for i in self.selected_]
        return self

    def transform(self, X):
        return X[:, self.selected_]


def mrmr(X: np.ndarray, y: np.ndarray, m: int, bins: int = 32) -> np.ndarray:
    """Greedy minimum-redundancy maximum-relevance (relevance: binned MI; redundancy: mean |corr|)."""
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rel = binned_mutual_information(X, y, bins)
    sd = Xc.std(axis=0)
    ok = sd > 0
    C = np.zeros((X.shape[1], X.shape[1]))
    if ok.sum() > 1:
        C[np.ix_(ok, ok)] = np.abs(np.corrcoef(Xc[:, ok], rowvar=False))
    C = np.nan_to_num(C, nan=0.0)
    chosen = [int(np.argmax(rel))]
    remaining = set(range(X.shape[1])) - set(chosen)
    while len(chosen) < min(m, X.shape[1]):
        rem = np.array(sorted(remaining))
        red = C[np.ix_(rem, chosen)].mean(axis=1)
        nxt = int(rem[np.argmax(rel[rem] / (red + 1e-3))])
        chosen.append(nxt)
        remaining.discard(nxt)
    return np.array(chosen)


class PCAmRMRSelector:
    """Hybrid: mRMR keeps 2k original features, PCA projects them onto k components."""

    def __init__(self, k: int = 40, seed: int = 0, max_rows: int | None = 300_000):
        self.k, self.seed, self.max_rows = k, seed, max_rows
        self.info = {"mode": f"mRMR({2 * k}) + PCA({k})", "max_rows": max_rows}

    def fit(self, X, y, w, names, **kw):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if self.max_rows and len(idx) > self.max_rows:
            idx = rng.choice(idx, size=self.max_rows, replace=False)
        self.cols_ = np.sort(mrmr(X[idx], np.asarray(y)[idx], min(2 * self.k, X.shape[1])))
        Xc = np.clip(np.nan_to_num(X[idx][:, self.cols_], nan=0.0, posinf=0.0, neginf=0.0), -30, 30)
        self.pca_ = PCA(n_components=min(self.k, len(self.cols_)), random_state=self.seed).fit(Xc)
        self.names_ = [f"mrmr_pc_{i:02d}" for i in range(self.pca_.n_components_)]
        return self

    def transform(self, X, chunk: int = 500_000):
        out = np.empty((len(X), self.pca_.n_components_), dtype=np.float32)
        for i in range(0, len(X), chunk):
            A = np.clip(np.nan_to_num(X[i: i + chunk][:, self.cols_], nan=0.0, posinf=0.0, neginf=0.0), -30, 30)
            out[i: i + chunk] = self.pca_.transform(A)
        return out


def make_selector(kind: str, k: int = 40, seed: int = 0, n_jobs: int = -1, **kw):
    """Selector factory.  ``kw`` carries the row cap (``max_rows``) and method-specific settings."""
    kind = kind.lower()
    if kind in ("proposed", "stability"):
        return StabilitySelector(rf_top=k, seed=seed, n_jobs=n_jobs, **kw)
    if kind == "none":
        return IdentitySelector()
    if kind in ("mi", "rf"):
        return TopKSelector(k, kind, seed, n_jobs=n_jobs, **kw)
    if kind == "pca":
        return PCASelector(k, seed, **kw)
    if kind == "rfe":
        return RFESelector(k, seed, n_jobs=n_jobs, **kw)
    if kind == "boruta":
        return BorutaSelector(k, seed, n_jobs=n_jobs, **kw)
    if kind == "ae":
        return AESelector(k, seed, **kw)
    if kind == "pca_mrmr":
        return PCAmRMRSelector(k, seed, **kw)
    raise ValueError(f"unknown selector '{kind}'")

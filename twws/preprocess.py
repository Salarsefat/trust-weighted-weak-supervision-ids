"""Step 4: cluster-median imputation and robust (median/IQR) scaling, plus the latent stage.

Everything here is fitted on training rows only and applied, frozen, to other rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from .features import NEVER_IMPUTE, LatentAE, feature_groups
from .utils import get_logger

log = get_logger("twws.preprocess")


class RobustImputerScaler:
    """Within-cluster median imputation followed by (x - median) / IQR scaling.

    * Flows are grouped with Mini-Batch K-means on their statistical sub-vector.
    * A missing entry of feature *j* is replaced by the median of feature *j* over the training
      members of the flow's cluster that have the feature observed (global training median if a
      cluster has no observed value).
    * Features listed in ``never_impute`` (payload entropy) are left missing; their binary
      missingness indicator is a separate feature.
    * Scaling uses training medians and inter-quartile ranges; a zero IQR is replaced by 1.
    """

    def __init__(self, n_clusters: int = 32, seed: int = 0, impute: bool = True, scale: bool = True,
                 never_impute: list[str] | None = None, fit_rows: int = 500_000) -> None:
        self.n_clusters, self.seed, self.impute, self.scale = n_clusters, seed, impute, scale
        self.never_impute = list(NEVER_IMPUTE if never_impute is None else never_impute)
        self.fit_rows = fit_rows
        self.info: dict = {}

    def fit(self, X: pd.DataFrame) -> "RobustImputerScaler":
        self.columns = list(X.columns)
        A = X.to_numpy(dtype=np.float64)
        self.median_ = np.nanmedian(A, axis=0)
        q1, q3 = np.nanpercentile(A, [25, 75], axis=0)
        iqr = q3 - q1
        self.iqr_ = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        self.stat_idx_ = [i for i, c in enumerate(self.columns) if not c.startswith(("t_", "z_"))]
        self.skip_idx_ = [i for i, c in enumerate(self.columns) if c in self.never_impute]
        self.miss_share_ = np.isnan(A).mean(axis=0)
        if self.impute:
            Z = self._cluster_space(A)
            rng = np.random.default_rng(self.seed)
            sub = rng.choice(len(Z), size=min(self.fit_rows, len(Z)), replace=False)
            k = int(min(self.n_clusters, max(2, len(sub) // 50)))
            self.km_ = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=self.seed).fit(Z[sub])
            lab = self._predict_clusters(Z)
            df = pd.DataFrame(A)
            df["_c"] = lab
            table = df.groupby("_c").median().reindex(range(k))
            T = table.to_numpy(dtype=np.float64)
            self.cluster_median_ = np.where(np.isfinite(T), T, self.median_[None, :])
        self.info = {"n_clusters": int(getattr(self, "km_", None).n_clusters) if self.impute else 0,
                     "impute": self.impute, "scale": self.scale, "never_impute": self.never_impute,
                     "features_with_missing": int((self.miss_share_ > 0).sum()),
                     "max_missing_share": float(self.miss_share_.max()) if len(self.miss_share_) else 0.0}
        return self

    def _cluster_space(self, A: np.ndarray) -> np.ndarray:
        Z = (A[:, self.stat_idx_] - self.median_[self.stat_idx_]) / self.iqr_[self.stat_idx_]
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0).astype(np.float32)

    def _predict_clusters(self, Z: np.ndarray, chunk: int = 500_000) -> np.ndarray:
        out = np.empty(len(Z), dtype=np.int32)
        for i in range(0, len(Z), chunk):
            out[i: i + chunk] = self.km_.predict(Z[i: i + chunk])
        return out

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        A = (X[self.columns].to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float64)).copy()
        if self.impute:
            lab = self._predict_clusters(self._cluster_space(A))
            miss = np.isnan(A)
            if self.skip_idx_:
                miss[:, self.skip_idx_] = False
            r, c = np.nonzero(miss)
            A[r, c] = self.cluster_median_[lab[r], c]
        if self.scale:
            A = (A - self.median_) / self.iqr_
        return A.astype(np.float32)


class FeaturePipeline:
    """Imputation/scaling of the base features and the autoencoder latent block.

    ``fit`` uses training rows only.  ``transform`` returns the full matrix
    ``[statistical | temporal | latent]`` (float32; NaN only in never-imputed columns).
    """

    def __init__(self, n_clusters: int = 32, seed: int = 0, impute: bool = True, scale: bool = True,
                 latent_dim: int = 32, ae_hidden: int = 64, ae_epochs: int = 30, ae_max_rows: int | None = 1_000_000,
                 use_latent: bool = True, threads: int | None = None) -> None:
        self.pre = RobustImputerScaler(n_clusters=n_clusters, seed=seed, impute=impute, scale=scale)
        self.use_latent = use_latent and latent_dim > 0
        self.ae = LatentAE(code=latent_dim, hidden=ae_hidden, max_epochs=ae_epochs, seed=seed, max_rows=ae_max_rows,
                           threads=threads) if self.use_latent else None
        self.info: dict = {}

    def fit(self, X_train: pd.DataFrame) -> "FeaturePipeline":
        self.base_columns = list(X_train.columns)
        groups = feature_groups(self.base_columns)
        self.stat_idx_ = [self.base_columns.index(c) for c in groups["statistical"]]
        self.pre.fit(X_train)
        A = self.pre.transform(X_train)
        if self.use_latent:
            self.ae.fit(A[:, self.stat_idx_])
            Z = self.ae.transform(A[:, self.stat_idx_])
            self.z_median_ = np.median(Z, axis=0)
            q1, q3 = np.percentile(Z, [25, 75], axis=0)
            self.z_iqr_ = np.where((q3 - q1) > 0, q3 - q1, 1.0)
            self.columns = self.base_columns + self.ae.names
        else:
            self.columns = list(self.base_columns)
        self.info = {"preprocessing": self.pre.info, "autoencoder": self.ae.info if self.use_latent else None,
                     "d_statistical": len(groups["statistical"]), "d_temporal": len(groups["temporal"]),
                     "d_latent": self.ae.code if self.use_latent else 0, "d_total": len(self.columns)}
        return self

    def transform(self, X: pd.DataFrame, chunk: int = 400_000) -> np.ndarray:
        out = np.empty((len(X), len(self.columns)), dtype=np.float32)
        nb = len(self.base_columns)
        for i in range(0, len(X), chunk):
            A = self.pre.transform(X.iloc[i: i + chunk])
            out[i: i + chunk, :nb] = A
            if self.use_latent:
                Z = self.ae.transform(A[:, self.stat_idx_])
                if self.pre.scale:
                    Z = (Z - self.z_median_) / self.z_iqr_
                out[i: i + chunk, nb:] = Z
        return out

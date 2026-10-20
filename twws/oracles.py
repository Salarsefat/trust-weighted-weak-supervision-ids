"""Step 2: the five weak-supervision oracles.

Every oracle returns one vote per flow: 1 (malicious), 0 (benign) or -1 (abstain).
No oracle uses ground-truth labels of the target corpus.  The transfer classifier is the
only component trained with labels, and those labels belong to a different dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .features import STAT_FEATURES
from .utils import get_logger

log = get_logger("twws.oracles")

MAL, BEN, ABS = 1, 0, -1
ORACLE_NAMES = ["stat", "signature", "cluster", "transfer", "domain"]
# two further label-free oracles, specified in docs/ORACLES_EXTENDED.md; they read the pair statistics of
# twws.pairs (aux_pairs.parquet of the corpus)
EXTENDED_ORACLES = ["rate", "beacon"]
ALL_ORACLES = ORACLE_NAMES + EXTENDED_ORACLES
ORACLE_LABELS = {"stat": "Statistical thresholds", "signature": "Signature-based IDS",
                 "cluster": "Clustering anomaly detection", "transfer": "Pre-trained transfer classifier",
                 "domain": "Domain heuristics", "rate": "Repeated connections to one service",
                 "beacon": "Regular repetition"}


# --------------------------------------------------------------------------- 1. statistical thresholds
@dataclass
class StatThresholdOracle:
    """Per-host, 1-minute window: SYN ratio above a threshold, or mean bytes/packet outside a band.

    A quantity within +-``guard`` (relative) of a threshold gives no evidence; if no rule is
    triggered and some quantity lies in a guard band, the oracle abstains.
    """
    syn_ratio: float = 0.45
    bpp_low: float = 45.0
    bpp_high: float = 1300.0
    guard: float = 0.05
    name: str = "stat"

    def vote(self, aux: pd.DataFrame) -> np.ndarray:
        syn = aux["aux_syn_ratio_1m"].to_numpy(dtype=np.float64)
        bpp = aux["aux_bpp_1m"].to_numpy(dtype=np.float64)
        g = self.guard
        with np.errstate(invalid="ignore"):
            trig = (syn > self.syn_ratio * (1 + g)) | (bpp < self.bpp_low * (1 - g)) | (bpp > self.bpp_high * (1 + g))
            in_guard = ((syn >= self.syn_ratio * (1 - g)) & (syn <= self.syn_ratio * (1 + g))) \
                | ((bpp >= self.bpp_low * (1 - g)) & (bpp <= self.bpp_low * (1 + g))) \
                | ((bpp >= self.bpp_high * (1 - g)) & (bpp <= self.bpp_high * (1 + g)))
        v = np.full(len(aux), BEN, dtype=np.int8)
        v[in_guard] = ABS
        v[trig] = MAL
        v[np.isnan(syn) & np.isnan(bpp)] = ABS
        return v

    def describe(self) -> dict:
        return {"syn_ratio": self.syn_ratio, "bpp_band": [self.bpp_low, self.bpp_high], "guard": self.guard}


# --------------------------------------------------------------------------- 2. signature IDS
@dataclass
class SignatureOracle:
    """Snort alert on any packet of the connection -> both directions malicious; else benign."""
    name: str = "signature"

    def vote(self, snort_alert: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(snort_alert, dtype=bool), MAL, BEN).astype(np.int8)

    def describe(self) -> dict:
        return {}


# --------------------------------------------------------------------------- 3. density clustering
@dataclass
class ClusterOracle:
    """DBSCAN point types on a PCA projection of the standardised statistical sub-vector.

    Only the point *type* matters for the vote, so the types are computed exactly from
    nearest-neighbour distances instead of materialising neighbourhoods:

    * core    at least ``min_samples`` training points (the point itself included) within eps
    * border  not core, but within eps of a core point
    * noise   otherwise

    core -> benign, noise -> malicious, border -> abstain.  For rows outside the training
    split the same definition is applied with respect to the training points.
    """
    eps: float = 1.6
    min_samples: int = 25
    n_components: int = 5
    seed: int = 0
    workers: int = -1
    name: str = "cluster"
    info: dict = field(default_factory=dict)

    def _project(self, S: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0))
        return self.pca_.transform(np.clip(Z, -50, 50))

    def fit(self, S_train: np.ndarray) -> "ClusterOracle":
        S = np.nan_to_num(np.asarray(S_train, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler().fit(S)
        k = min(self.n_components, S.shape[1])
        self.pca_ = PCA(n_components=k, random_state=self.seed).fit(np.clip(self.scaler_.transform(S), -50, 50))
        P = self.pca_.transform(np.clip(self.scaler_.transform(S), -50, 50))
        self.tree_ = cKDTree(P)
        kk = min(self.min_samples, len(P))
        d, _ = self.tree_.query(P, k=kk, workers=self.workers)
        d = d if d.ndim == 1 else d[:, -1]
        core = d <= self.eps
        self.core_tree_ = cKDTree(P[core]) if core.any() else None
        self.train_types_ = self._types(P, core)
        self.info = {"eps": self.eps, "min_samples": self.min_samples, "pca_components": int(k),
                     "pca_explained_variance": float(self.pca_.explained_variance_ratio_.sum()),
                     "train_core_share": float(core.mean()),
                     "train_border_share": float((self.train_types_ == 1).mean()),
                     "train_noise_share": float((self.train_types_ == 2).mean())}
        return self

    def _types(self, P: np.ndarray, core: np.ndarray) -> np.ndarray:
        """0 core, 1 border, 2 noise."""
        t = np.full(len(P), 2, dtype=np.int8)
        t[core] = 0
        rest = np.flatnonzero(~core)
        if self.core_tree_ is not None and len(rest):
            d, _ = self.core_tree_.query(P[rest], k=1, distance_upper_bound=self.eps, workers=self.workers)
            t[rest[np.isfinite(d)]] = 1
        return t

    def types(self, S: np.ndarray, is_train: np.ndarray | None = None) -> np.ndarray:
        P = self._project(np.asarray(S, dtype=np.float64))
        kk = min(self.min_samples, self.tree_.n)
        d, _ = self.tree_.query(P, k=kk, workers=self.workers)
        if d.ndim == 1:
            d = d[:, None]
        # a row outside the training set counts itself, so it needs min_samples - 1 training neighbours
        kth_out = d[:, -2] if d.shape[1] >= 2 else d[:, -1]
        kth_in = d[:, -1]
        kth = kth_out if is_train is None else np.where(np.asarray(is_train, dtype=bool), kth_in, kth_out)
        return self._types(P, kth <= self.eps)

    def vote(self, S: np.ndarray, is_train: np.ndarray | None = None) -> np.ndarray:
        t = self.types(S, is_train)
        return np.where(t == 0, BEN, np.where(t == 2, MAL, ABS)).astype(np.int8)

    def describe(self) -> dict:
        return dict(self.info)


# --------------------------------------------------------------------------- 4. transfer classifier
@dataclass
class TransferOracle:
    """Random forest trained on a *different* labelled dataset, applied to per-flow statistics."""
    hi: float = 0.6
    lo: float = 0.4
    name: str = "transfer"
    model: RandomForestClassifier | None = None
    info: dict = field(default_factory=dict)
    columns: list[str] = field(default_factory=lambda: list(STAT_FEATURES))

    @staticmethod
    def _prep(S: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(S, dtype=np.float32), nan=-1.0, posinf=-1.0, neginf=-1.0)

    def fit_source(self, S_src: np.ndarray, y_src: np.ndarray, n_estimators: int = 120, max_depth: int = 16,
                   seed: int = 0, n_jobs: int = -1, holdout: float = 0.2) -> "TransferOracle":
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_src))
        n_ho = int(holdout * len(idx))
        ho, tr = idx[:n_ho], idx[n_ho:]
        X = self._prep(S_src)
        probe = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
                                       random_state=seed, class_weight="balanced_subsample").fit(X[tr], y_src[tr])
        pred = probe.predict(X[ho])
        p, r, f, _ = precision_recall_fscore_support(y_src[ho], pred, average="binary", zero_division=0)
        self.info = {"source_rows": int(len(y_src)), "source_malicious_share": float(np.mean(y_src)),
                     "source_holdout_accuracy": float(accuracy_score(y_src[ho], pred)),
                     "source_holdout_precision": float(p), "source_holdout_recall": float(r),
                     "source_holdout_f1": float(f), "n_estimators": n_estimators, "max_depth": max_depth,
                     "holdout": holdout, "features": list(self.columns)}
        # the deployed oracle is the forest fitted on the training part of the source data
        self.model = probe
        return self

    def proba(self, S: np.ndarray, chunk: int = 500_000) -> np.ndarray:
        X = self._prep(S)
        out = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), chunk):
            out[i: i + chunk] = self.model.predict_proba(X[i: i + chunk])[:, 1]
        return out

    def vote(self, S: np.ndarray) -> np.ndarray:
        p = self.proba(S)
        return np.where(p > self.hi, MAL, np.where(p < self.lo, BEN, ABS)).astype(np.int8)

    def describe(self) -> dict:
        return {"hi": self.hi, "lo": self.lo, **self.info}


# --------------------------------------------------------------------------- 5. domain heuristics
@dataclass
class DomainOracle:
    """Threat-intelligence list plus an anomalous-TLS / fan-out rule.

    malicious  an endpoint is on the C&C list; or the leaf certificate is self-signed and the
               service port is above ``tls_port_min`` (both directions of the connection); or
               the flow is the originator direction of a connection whose host, as originator,
               contacted at least ``fanout_min`` distinct destination addresses during the
               last 5 minutes, the current flow included (the reply direction of such a
               connection is NOT flagged by this rule);
    benign     a TLS handshake was observed and no indicator fired;
    abstain    otherwise (no list hit and no TLS metadata).
    """
    c2_addresses: frozenset = frozenset()
    tls_port_min: int = 8000
    fanout_min: float = 15
    name: str = "domain"

    def vote(self, meta: pd.DataFrame, aux: pd.DataFrame) -> np.ndarray:
        n = len(meta)
        listed = np.zeros(n, dtype=bool)
        if self.c2_addresses:
            listed = (meta["src_ip"].astype(str).isin(self.c2_addresses)
                      | meta["dst_ip"].astype(str).isin(self.c2_addresses)).to_numpy()
        is_orig = meta["dir"].to_numpy() == 0
        service_port = np.where(is_orig, meta["dst_port"].to_numpy(), meta["src_port"].to_numpy())
        selfs = meta["tls_self_signed"].to_numpy(dtype=np.float64)
        tls = meta["tls"].to_numpy() > 0
        bad_tls = (selfs == 1.0) & (service_port > self.tls_port_min)
        fan = is_orig & (aux["aux_fanout_5m"].to_numpy(dtype=np.float64) >= self.fanout_min)
        v = np.full(n, ABS, dtype=np.int8)
        v[tls] = BEN
        v[listed | bad_tls | fan] = MAL
        self.last_counts = {"listed": int(listed.sum()), "self_signed_high_port": int(bad_tls.sum()),
                            "fanout": int(fan.sum())}
        return v

    def describe(self) -> dict:
        return {"c2_list_size": len(self.c2_addresses), "tls_port_min": self.tls_port_min,
                "fanout_min": self.fanout_min}


# --------------------------------------------------------------------------- 6. repeated connections
@dataclass
class RateOracle:
    """Repeated TCP connections to one service of one host (docs/ORACLES_EXTENDED.md, oracle 6).

    Quantity: ``aux_pair_n60``, the connections of the pair (originator, responder, responder port)
    that started in the last 60 s, the current one included.  Threshold: a quantile, over the pairs
    of the reference (training) rows, of each pair's largest value, so that a flood, which is one
    pair however many flows it has, does not move it.  malicious above T(1+guard), abstain within
    the guard band, benign below; abstain for non-TCP flows (no pair).
    """
    threshold: float = float("nan")
    guard: float = 0.05
    name: str = "rate"

    @staticmethod
    def pair_threshold(pair_id: np.ndarray, n60: np.ndarray, ref: np.ndarray, quantile: float) -> float:
        pair_id = np.asarray(pair_id, dtype=np.int64)
        n60 = np.asarray(n60, dtype=np.float64)
        ok = np.asarray(ref, dtype=bool) & (pair_id >= 0) & np.isfinite(n60)
        if not ok.any():
            return float("nan")
        peak = pd.Series(n60[ok]).groupby(pair_id[ok]).max().to_numpy()
        return float(np.quantile(peak, float(quantile)))

    def vote(self, aux: pd.DataFrame) -> np.ndarray:
        n = aux["aux_pair_n60"].to_numpy(dtype=np.float64)
        T, g = float(self.threshold), float(self.guard)
        v = np.full(len(n), ABS, dtype=np.int8)
        ok = np.isfinite(n) & np.isfinite(T)
        v[ok & (n < T * (1 - g))] = BEN
        v[ok & (n > T * (1 + g))] = MAL
        return v

    def describe(self) -> dict:
        return {"threshold": self.threshold, "guard": self.guard}


# --------------------------------------------------------------------------- 7. regular repetition
@dataclass
class BeaconOracle:
    """Regular repetition of TCP connections of one pair (docs/ORACLES_EXTENDED.md, oracle 7).

    Over the connections of the pair that started in the last hour (``aux_pair_n1h``, at least
    ``min_conn``) with a median inter-arrival time ``aux_pair_iat_med`` of at least ``iat_min``:
    malicious if the dispersion ``aux_pair_iat_disp`` (MAD / median) is at most ``disp_max``, benign
    if it exceeds ``benign_factor * disp_max``, abstain in between and in every other case.
    """
    min_conn: int = 10
    disp_max: float = 0.1
    iat_min: float = 1.0
    benign_factor: float = 2.0
    name: str = "beacon"

    def vote(self, aux: pd.DataFrame) -> np.ndarray:
        n1h = aux["aux_pair_n1h"].to_numpy(dtype=np.float64)
        med = aux["aux_pair_iat_med"].to_numpy(dtype=np.float64)
        disp = aux["aux_pair_iat_disp"].to_numpy(dtype=np.float64)
        v = np.full(len(n1h), ABS, dtype=np.int8)
        with np.errstate(invalid="ignore"):
            enough = np.isfinite(n1h) & (n1h >= self.min_conn) & np.isfinite(med) & (med >= self.iat_min)
            v[enough & (disp > self.benign_factor * self.disp_max)] = BEN
            v[enough & (disp <= self.disp_max)] = MAL
        return v

    def describe(self) -> dict:
        return {"min_conn": self.min_conn, "disp_max": self.disp_max, "iat_min": self.iat_min,
                "benign_disp_above": self.benign_factor * self.disp_max}


def load_c2_list(path: str | None) -> frozenset:
    """Read an abuse.ch Feodo Tracker blocklist (CSV or plain list); '#' lines are comments."""
    import ipaddress
    import re

    if not path:
        return frozenset()
    out = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            for tok in re.split(r"[,\s\"]+", line.strip()):
                try:
                    ipaddress.ip_address(tok)
                    out.add(tok)
                except ValueError:
                    continue
    return frozenset(out)

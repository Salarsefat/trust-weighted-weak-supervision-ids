"""End-to-end preparation of one experimental condition (Algorithm 1, lines 2-6).

``prepare`` turns the cached corpus into everything a detector needs:

    stage A   partition, imputation/scaling, autoencoder, full feature matrix
    stage B   votes of the five oracles (fitted on training rows only)
    stage C   majority vote and trust, feature selection, semi-supervised refinement

Stages A and B are memoised in memory so that ablations which only change stage C or the
detector re-use them.  Ground-truth labels are carried along for evaluation only; nothing in
this module reads them for fitting.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .corpus import TEST, TRAIN, VAL_A, VAL_B, Workspace, make_split, split_table
from .features import STAT_FEATURES, feature_groups
from .fselect import make_selector
from .oracles import (EXTENDED_ORACLES, ORACLE_NAMES, BeaconOracle, ClusterOracle, DomainOracle, RateOracle,
                      SignatureOracle, StatThresholdOracle, TransferOracle, load_c2_list)
from .preprocess import FeaturePipeline
from .utils import get_logger
from .weak import NO_MAJORITY, aggregate_votes, high_trust_mask, refine

log = get_logger("twws.pipeline")


@dataclass(frozen=True)
class PrepOptions:
    protocol: str = "random"
    split_seed: int = 42
    seed: int = 0
    forward: tuple | None = None                  # (("train", (...days)), ("val", ...), ("test", ...))
    # stage A
    impute: bool = True
    scale: bool = True
    use_latent: bool = True
    latent_dim: int = 32
    temporal: str = "global"                      # 'global' or 'none' (drops all host-level features)
    n_clusters: int = 32
    # stage B
    stat_syn_ratio: float = 0.45
    stat_bpp_low: float = 45.0
    stat_bpp_high: float = 1300.0
    stat_guard: float = 0.05
    cluster_eps: float = 1.6
    cluster_min_samples: int = 25
    cluster_pca: int = 5
    transfer_hi: float = 0.6
    transfer_lo: float = 0.4
    domain_tls_port_min: int = 8000
    domain_fanout_min: float = 15
    domain_fanout_quantile: float | None = None   # threshold = this quantile of the training originator flows
    signature_any: bool = False                   # True: every Snort alert counts (definition of the first version)
    # oracles 'rate' and 'beacon' (docs/ORACLES_EXTENDED.md); used only when they are in ``oracles``
    rate_quantile: float = 0.999                  # threshold = this quantile over the pairs of the training rows
    rate_guard: float = 0.05
    beacon_min_conn: int = 10
    beacon_disp_max: float = 0.1
    beacon_iat_min: float = 1.0
    # stage C
    oracles: tuple = tuple(ORACLE_NAMES)
    aggregation: str = "ds"                       # 'ds' (label model), 'ds_grouped' (dependent oracles merged,
                                                  # docs/ORACLES_EXTENDED.md) or 'majority' (first submission)
    group_jaccard: float = 0.5                    # ds_grouped: Jaccard index of malicious votes that merges oracles
    selector: str = "proposed"
    k: int = 40
    theta: float = 0.8
    mi_top: int = 80
    n_folds: int = 5
    use_borderline: bool = True
    refine: bool = True
    p_flip: float = 0.85
    refined_trust: float = 0.6
    high_trust: float = 0.8

    def key(self, fields: tuple[str, ...] | None = None) -> str:
        d = asdict(self)
        if fields is not None:
            d = {k: d[k] for k in fields}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


STAGE_A_FIELDS = ("protocol", "split_seed", "seed", "forward", "impute", "scale", "use_latent", "latent_dim",
                  "temporal", "n_clusters")
STAGE_A_SPLIT_FIELDS = ("protocol", "split_seed", "forward")
# options that each oracle's votes depend on (the votes are memoised on exactly these)
ORACLE_DEPENDENCIES = {
    "stat": ("stat_syn_ratio", "stat_bpp_low", "stat_bpp_high", "stat_guard"),
    "signature": ("signature_any",),
    "cluster": STAGE_A_SPLIT_FIELDS + ("seed", "cluster_eps", "cluster_min_samples", "cluster_pca"),
    "transfer": ("transfer_hi", "transfer_lo"),
    "domain": STAGE_A_SPLIT_FIELDS + ("domain_tls_port_min", "domain_fanout_min", "domain_fanout_quantile"),
    "rate": STAGE_A_SPLIT_FIELDS + ("rate_quantile", "rate_guard"),
    "beacon": ("beacon_min_conn", "beacon_disp_max", "beacon_iat_min"),
}
# stage B (all votes) is cached on the stage-A options and every option that an oracle depends on.  An option that
# is missing here makes a variant silently reuse the votes of another configuration (tests/test_pipeline_cache.py).
STAGE_B_FIELDS = STAGE_A_FIELDS + tuple(dict.fromkeys(
    f for deps in ORACLE_DEPENDENCIES.values() for f in deps if f not in STAGE_A_FIELDS))


@dataclass
class Prepared:
    opts: PrepOptions
    X: np.ndarray                       # (n, d) float32, all features
    names: list
    split: np.ndarray
    y_true: np.ndarray                  # evaluation only
    family: np.ndarray                  # evaluation only
    votes: np.ndarray                   # (n, K) for the oracles in ``oracle_names``
    oracle_names: list
    agg: dict
    y_pl: np.ndarray                    # pseudo-label after refinement
    w_pl: np.ndarray                    # training weight (trust)
    refined: np.ndarray
    selected: np.ndarray                # column indices into X (or None when the selector transforms)
    sel_names: list
    selector: object = None
    info: dict = field(default_factory=dict)
    feature_pipeline: object = None
    meta_ref: object = None             # corpus metadata (addresses, times) for sequence/graph models
    _Xs: np.ndarray | None = None

    @property
    def train(self) -> np.ndarray:
        return self.split == TRAIN

    @property
    def val_a(self) -> np.ndarray:
        return self.split == VAL_A

    @property
    def val_b(self) -> np.ndarray:
        return self.split == VAL_B

    @property
    def test(self) -> np.ndarray:
        return self.split == TEST

    @property
    def Xs(self) -> np.ndarray:
        """Feature matrix restricted to / transformed by the selector."""
        if self._Xs is None:
            self._Xs = np.ascontiguousarray(self.selector.transform(self.X), dtype=np.float32)
        return self._Xs

    def usable_train(self) -> np.ndarray:
        return self.train & (self.w_pl > 0)


class Preparer:
    """Runs and memoises the preparation stages for one workspace."""

    def __init__(self, ws: Workspace, cfg, n_jobs: int | None = None) -> None:
        self.ws, self.cfg = ws, cfg
        self.n_jobs = n_jobs or int(cfg.get_path("compute.n_jobs", -1))
        self._A: dict = {}
        self._B: dict = {}
        self._C: dict = {}
        self._V: dict = {}                      # per-oracle votes
        self.max_cache = int(cfg.get_path("compute.prep_cache", 2))
        self._transfer = None

    # ------------------------------------------------------------------ helpers
    def transfer_model(self) -> TransferOracle | None:
        if self._transfer is None:
            p = Path(self.cfg.get_path("transfer.model_path") or (self.ws.root / "transfer" / "transfer_rf.joblib"))
            if p.exists():
                self._transfer = joblib.load(p)
        return self._transfer

    def available_oracles(self) -> list[str]:
        names = list(self.cfg.get_path("oracles.enabled", ORACLE_NAMES))
        unknown = [n for n in names if n not in ORACLE_DEPENDENCIES]
        if unknown:
            raise ValueError(f"unknown oracle(s) in oracles.enabled: {unknown}")
        if any(n in EXTENDED_ORACLES for n in names) and not self.ws.has_pairs:
            raise FileNotFoundError(
                f"the oracles {[n for n in names if n in EXTENDED_ORACLES]} need the pair statistics of the corpus "
                f"({self.ws.pairs_path}): run `twws.cli pairs`")
        if "transfer" in names and self.transfer_model() is None:
            if bool(self.cfg.get_path("oracles.allow_missing_transfer", False)):
                log.warning("transfer oracle requested but no source model found; it is left out "
                            "(oracles.allow_missing_transfer is set)")
                names.remove("transfer")
            else:
                raise FileNotFoundError(
                    "the transfer oracle is enabled (oracles.enabled) but its source model does not exist: build the "
                    "second corpus and run `twws.cli transfer`, or remove 'transfer' from oracles.enabled and say so "
                    "in the paper (the experiments then run with four oracles)")
        return names

    def _trim(self, cache: dict) -> None:
        while len(cache) > self.max_cache:
            cache.pop(next(iter(cache)))

    @staticmethod
    def _hit(cache: dict, key):
        """Least-recently-used behaviour: a cache hit moves the entry to the end."""
        val = cache.pop(key)
        cache[key] = val
        return val

    # ------------------------------------------------------------------ stage A
    def stage_a(self, o: PrepOptions) -> dict:
        key = o.key(STAGE_A_FIELDS)
        if key in self._A:
            return self._hit(self._A, key)
        t0 = time.perf_counter()
        meta, Xb = self.ws.meta, self.ws.X_base
        forward = {k: list(v) for k, v in o.forward} if o.forward else None
        split = make_split(meta, o.protocol, o.split_seed, forward)
        cols = list(Xb.columns)
        if o.temporal == "none":
            cols = [c for c in cols if not c.startswith(("t_", "h_"))]
        fp = FeaturePipeline(n_clusters=o.n_clusters, seed=o.seed, impute=o.impute, scale=o.scale,
                             latent_dim=o.latent_dim, use_latent=o.use_latent,
                             ae_epochs=int(self.cfg.get_path("features.ae_epochs", 30)),
                             ae_hidden=int(self.cfg.get_path("features.ae_hidden", 64)),
                             ae_max_rows=self.cfg.get_path("features.ae_max_rows", 1_000_000),
                             threads=None if self.n_jobs in (None, -1) else self.n_jobs)
        fp.fit(Xb.loc[split == TRAIN, cols])
        X = fp.transform(Xb[cols])
        out = {"split": split, "X": X, "names": list(fp.columns), "fp": fp, "cols": cols,
               "seconds": time.perf_counter() - t0, "split_table": split_table(meta, split)}
        self._A[key] = out
        self._trim(self._A)
        log.info("stage A done in %.0f s: X %s", out["seconds"], X.shape)
        return out

    # ------------------------------------------------------------------ stage B
    def _oracle_votes(self, name: str, o: PrepOptions, A: dict) -> dict:
        """Votes of one oracle, memoised on exactly the options that oracle depends on."""
        dep = ORACLE_DEPENDENCIES[name]
        key = (name, o.key(dep))
        if key in self._V:
            return self._V[key]
        meta, Xb, aux = self.ws.meta, self.ws.X_base, self.ws.aux
        t = time.perf_counter()
        if name == "stat":
            orc = StatThresholdOracle(o.stat_syn_ratio, o.stat_bpp_low, o.stat_bpp_high, o.stat_guard)
            votes, desc = orc.vote(aux), orc.describe()
        elif name == "signature":
            col = "snort_alert_any" if o.signature_any else "snort_alert"
            if col not in meta.columns:
                raise KeyError(f"the corpus has no column {col!r}: rebuild it with `twws.cli corpus`")
            orc = SignatureOracle()
            votes = orc.vote(meta[col].to_numpy())
            desc = {**orc.describe(), "alerts": "every alert" if o.signature_any else "attack classes"}
        elif name == "cluster":
            stat_cols = feature_groups(list(Xb.columns))["statistical"]
            S = Xb[stat_cols].to_numpy(dtype=np.float32)
            fit_idx = np.flatnonzero(A["split"] == TRAIN)
            cap = self.cfg.get_path("oracles.cluster_max_fit_rows")
            if cap and len(fit_idx) > cap:
                fit_idx = np.sort(np.random.default_rng(o.seed).choice(fit_idx, size=int(cap), replace=False))
            fit_flag = np.zeros(len(S), dtype=bool)
            fit_flag[fit_idx] = True
            orc = ClusterOracle(eps=o.cluster_eps, min_samples=o.cluster_min_samples, n_components=o.cluster_pca,
                                seed=o.seed, workers=self.n_jobs).fit(S[fit_idx])
            votes = orc.vote(S, is_train=fit_flag)
            desc = {**orc.describe(), "fit_rows": int(len(fit_idx))}
        elif name == "transfer":
            orc = self.transfer_model()
            orc.hi, orc.lo = o.transfer_hi, o.transfer_lo
            votes, desc = orc.vote(Xb[STAT_FEATURES].to_numpy(dtype=np.float32)), orc.describe()
        elif name == "domain":
            thr = float(o.domain_fanout_min)
            if o.domain_fanout_quantile is not None:     # quantile of the fan-out of the training originator flows
                fan = aux["aux_fanout_5m"].to_numpy(dtype=np.float64)
                ref = (A["split"] == TRAIN) & (meta["dir"].to_numpy() == 0) & np.isfinite(fan)
                thr = float(np.quantile(fan[ref], float(o.domain_fanout_quantile))) if ref.any() else thr
            orc = DomainOracle(load_c2_list(self.cfg.get_path("oracles.c2_list")), o.domain_tls_port_min, thr)
            votes = orc.vote(meta, aux)
            desc = {**orc.describe(), **orc.last_counts, "fanout_quantile": o.domain_fanout_quantile,
                    "fanout_threshold": thr}
        elif name == "rate":
            ref = A["split"] == TRAIN
            thr = RateOracle.pair_threshold(aux["aux_pair_id"].to_numpy(), aux["aux_pair_n60"].to_numpy(), ref,
                                            o.rate_quantile)
            orc = RateOracle(threshold=thr, guard=o.rate_guard)
            votes = orc.vote(aux)
            desc = {**orc.describe(), "quantile": o.rate_quantile,
                    "reference_pairs": int(aux.loc[ref & (aux["aux_pair_id"].to_numpy() >= 0), "aux_pair_id"].nunique())}
        elif name == "beacon":
            orc = BeaconOracle(min_conn=int(o.beacon_min_conn), disp_max=float(o.beacon_disp_max),
                               iat_min=float(o.beacon_iat_min))
            votes, desc = orc.vote(aux), orc.describe()
        else:  # pragma: no cover
            raise ValueError(name)
        out = {"votes": votes.astype(np.int8), "describe": desc, "seconds": time.perf_counter() - t}
        self._V[key] = out
        while len(self._V) > 64:
            self._V.pop(next(iter(self._V)))
        return out

    def stage_b(self, o: PrepOptions) -> dict:
        key = o.key(STAGE_B_FIELDS)
        if key in self._B:
            return self._hit(self._B, key)
        A = self.stage_a(o)
        t0 = time.perf_counter()
        names = self.available_oracles()
        votes, desc, timing = {}, {}, {}
        for n in names:
            r = self._oracle_votes(n, o, A)
            votes[n], desc[n], timing[n] = r["votes"], r["describe"], r["seconds"]
        out = {"votes": votes, "describe": desc, "names": names, "timing": timing,
               "seconds": time.perf_counter() - t0}
        self._B[key] = out
        self._trim(self._B)
        log.info("stage B done in %.0f s: oracles %s", out["seconds"], names)
        return out

    # ------------------------------------------------------------------ stage C
    def prepare(self, o: PrepOptions | None = None, **changes) -> Prepared:
        o = o or PrepOptions()
        if changes:
            o = PrepOptions(**{**asdict(o), **changes})
        key = o.key()
        if key in self._C:
            return self._hit(self._C, key)
        A, B = self.stage_a(o), self.stage_b(o)
        t0 = time.perf_counter()
        meta = self.ws.meta
        split, X, names = A["split"], A["X"], A["names"]
        use = [n for n in o.oracles if n in B["names"]]
        if not use:
            raise RuntimeError("no oracle available")
        votes = np.column_stack([B["votes"][n] for n in use]).astype(np.int8)
        is_train = split == TRAIN
        agg = aggregate_votes(votes, o.aggregation, fit_mask=is_train,  # label model sees training votes only
                              names=use, group_jaccard=o.group_jaccard)
        has_major = agg["label"] != NO_MAJORITY
        high = high_trust_mask(agg, o.high_trust)
        # feature selection on training rows that have a majority label
        fs_rows = np.flatnonzero(is_train & has_major)
        sel_kw = {}
        if o.selector in ("proposed", "stability"):
            sel_kw = {"theta": o.theta, "mi_top": o.mi_top, "n_folds": o.n_folds, "use_borderline": o.use_borderline,
                      "borderline_low": float(self.cfg.get_path("selection.borderline_low", 0.5)),
                      "max_rows": self.cfg.get_path("selection.max_rows", 600_000),
                      "rf_max_samples": self.cfg.get_path("selection.rf_max_samples", 200_000)}
        else:
            cap = self.cfg.get_path("selection.compare_max_rows")
            if cap and o.selector in ("mi", "rf", "pca", "rfe", "boruta", "ae", "pca_mrmr"):
                sel_kw["max_rows"] = int(cap)
            if o.selector == "boruta":
                sel_kw["max_iter"] = int(self.cfg.get_path("selection.boruta_max_iter", 30))
        selector = make_selector(o.selector, k=o.k, seed=o.seed, n_jobs=self.n_jobs, **sel_kw)
        t_fs = time.perf_counter()
        selector.fit(X[fs_rows], agg["label"][fs_rows], agg["trust"][fs_rows], names, high_trust=high[fs_rows])
        fs_seconds = time.perf_counter() - t_fs
        sel_idx = getattr(selector, "selected_", None)
        prep = Prepared(opts=o, X=X, names=names, split=split, y_true=(meta["label"].to_numpy() == 1).astype(np.int8),
                        family=meta["family"].astype(str).to_numpy(), votes=votes, oracle_names=use, agg=agg,
                        y_pl=agg["label"], w_pl=agg["trust"], refined=np.zeros(len(split), bool),
                        selected=sel_idx, sel_names=list(selector.names_), selector=selector, feature_pipeline=A["fp"],
                        meta_ref=meta)
        t_ref = time.perf_counter()
        ref = refine(prep.Xs, agg, fit_mask=is_train,
                     cfg={"enabled": o.refine, "p_flip": o.p_flip, "refined_trust": o.refined_trust,
                          "high_trust": o.high_trust,
                          "max_fit_rows": self.cfg.get_path("refinement.max_fit_rows", 500_000)}, seed=o.seed)
        prep.y_pl, prep.w_pl, prep.refined = ref["label"], ref["weight"], ref["refined"]
        train_ref = ref["refined"] & is_train
        prep.info = {
            "options": asdict(o), "oracles": use, "oracle_describe": {n: B["describe"][n] for n in use},
            "feature_pipeline": A["fp"].info, "split_table": A["split_table"],
            "selection": {**getattr(selector, "info", {}), "selected": list(selector.names_),
                          "seconds": fs_seconds},
            "refinement": {**ref["info"], "train_low_trust": int(train_ref.sum()),
                           "train_relabelled": int((train_ref & (ref["label"] != agg["label"]) & has_major).sum()),
                           "seconds": time.perf_counter() - t_ref},
            "timing": {"stage_a": A["seconds"], "stage_b": B["seconds"], "oracles": B["timing"],
                       "stage_c": time.perf_counter() - t0},
            "n_rows": int(len(split)), "n_features": int(X.shape[1]), "n_selected": int(prep.Xs.shape[1]),
        }
        self._C[key] = prep
        self._trim(self._C)
        log.info("prepared %s/%s: %d selected features, %d low-trust training rows", o.protocol, o.selector,
                 prep.Xs.shape[1], int(train_ref.sum()))
        return prep

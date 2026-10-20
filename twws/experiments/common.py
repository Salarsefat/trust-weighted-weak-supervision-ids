"""Shared context and helpers for the experiment modules."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

from ..config import REPO_ROOT
from ..corpus import Workspace
from ..metrics import (binary_metrics, bootstrap_counts_ci, per_family_recall, seed_summary,
                       summarise_runs)
from ..model import DEFAULT_PARAMS, calibrate_threshold, train_twxgb
from ..pipeline import PrepOptions, Preparer
from ..stages import workspace
from ..utils import environment_record, get_logger

log = get_logger("twws.exp")


class Ctx:
    """Configuration, workspace and the (memoising) preparer shared by all experiments."""

    def __init__(self, cfg, n_jobs: int | None = None) -> None:
        self.cfg = cfg
        self.ws: Workspace = workspace(cfg)
        nj = int(cfg.get_path("compute.n_jobs", -1)) if n_jobs is None else n_jobs
        self.n_jobs = (os.cpu_count() or 1) if nj in (-1, None) else nj
        self.prep = Preparer(self.ws, cfg, n_jobs=self.n_jobs)
        self._best = None
        self.params_source = None

    # ------------------------------------------------------------------ options
    def base_options(self, **changes) -> PrepOptions:
        c = self.cfg
        o = PrepOptions(
            seed=int(c.experiments.seeds[0]),
            latent_dim=int(c.features.latent_dim), n_clusters=int(c.features.impute_clusters),
            stat_syn_ratio=float(c.oracles.stat.syn_ratio), stat_bpp_low=float(c.oracles.stat.bpp_low),
            stat_bpp_high=float(c.oracles.stat.bpp_high), stat_guard=float(c.oracles.stat.guard),
            cluster_eps=float(c.oracles.cluster.eps), cluster_min_samples=int(c.oracles.cluster.min_samples),
            cluster_pca=int(c.oracles.cluster.pca_components), transfer_hi=float(c.oracles.transfer.hi),
            transfer_lo=float(c.oracles.transfer.lo), domain_tls_port_min=int(c.oracles.domain.tls_port_min),
            domain_fanout_min=float(c.oracles.domain.fanout_min),
            domain_fanout_quantile=(None if c.oracles.domain.get("fanout_quantile") is None
                                    else float(c.oracles.domain.fanout_quantile)),
            signature_any=bool(c.oracles.get("signature_any_alert", False)),
            rate_quantile=float(c.get_path("oracles.rate.quantile", 0.999)),
            rate_guard=float(c.get_path("oracles.rate.guard", 0.05)),
            beacon_min_conn=int(c.get_path("oracles.beacon.min_conn", 10)),
            beacon_disp_max=float(c.get_path("oracles.beacon.disp_max", 0.1)),
            beacon_iat_min=float(c.get_path("oracles.beacon.iat_min", 1.0)),
            oracles=tuple(self.prep.available_oracles()),
            aggregation=str(c.get_path("oracles.aggregation", "ds")),
            group_jaccard=float(c.get_path("oracles.group_jaccard", 0.5)),
            k=int(c.selection.k), theta=float(c.selection.theta), mi_top=int(c.selection.mi_top),
            n_folds=int(c.selection.n_folds), refine=bool(c.get_path("refinement.enabled", True)),
            p_flip=float(c.refinement.p_flip),
            refined_trust=float(c.refinement.refined_trust), high_trust=float(c.refinement.high_trust))
        if changes:
            o = PrepOptions(**{**asdict(o), **changes})
        return o

    def seeds(self, kind: str = "main") -> list[int]:
        """Model seeds: 'main' (headline result, baselines), 'secondary' (protocols, LOAO,
        selection comparison, cross-dataset) or 'ablation'."""
        e = self.cfg.experiments
        if kind == "secondary":
            return [int(x) for x in (e.get("secondary_seeds") or e.seeds)]
        if kind == "ablation":
            return [int(x) for x in e.ablation_seeds]
        return [int(x) for x in e.seeds]

    def default_prep(self):
        return self.prep.prepare(self.base_options())

    # ------------------------------------------------------------------ detector parameters
    def best_params(self) -> dict:
        """Hyper-parameters selected by the search.

        Without a search result the experiments stop, unless ``experiments.allow_default_params`` is set
        (code tests only): results must never be produced with unselected defaults without saying so.
        """
        if self._best is None:
            p = {**DEFAULT_PARAMS, **dict(self.cfg.model.defaults)}
            src = self.cfg.get_path("search.params_from")
            if self.ws.has_result("search"):
                p.update(self.ws.load_result("search")["best_params"])
                self.params_source = "search"
            elif src:
                # hyperparameters selected in another workspace (configs/ids2017_ext.yaml: those of the search on
                # CSE-CIC-IDS2018); the file must exist, nothing falls back to defaults
                path = Path(src) if Path(src).is_absolute() else REPO_ROOT / src
                if not path.exists():
                    raise RuntimeError(f"search.params_from: {path} does not exist")
                p.update(json.loads(path.read_text(encoding="utf-8"))["best_params"])
                self.params_source = f"search of another workspace ({src})"
            elif bool(self.cfg.experiments.get("allow_default_params", False)):
                log.warning("no search result: using model.defaults (experiments.allow_default_params is set)")
                self.params_source = "defaults"
            else:
                raise RuntimeError("no hyperparameter search result in this workspace: run `twws.cli run search` first "
                                   "(or set experiments.allow_default_params=true for code tests)")
            if p.get("class_weight") is None:          # the search hands the unweighted option over as None
                p["class_weight"] = "none"
            self._best = p
        return dict(self._best)

    def reset_best(self) -> None:
        self._best = None


def run_detector(prep, params: dict, seed: int, cfg, n_jobs: int, keep_scores: bool = False,
                 train_rows: np.ndarray | None = None, y_train: np.ndarray | None = None,
                 w_train: np.ndarray | None = None, truth_for_validation: bool = False) -> dict:
    """Train on pseudo-labels, calibrate the threshold on pseudo-labels, evaluate on ground truth.

    ``truth_for_validation`` (supervised reference only) uses ground-truth labels for early
    stopping and threshold calibration as well.
    """
    tr = np.flatnonzero(prep.usable_train()) if train_rows is None else train_rows
    y = prep.y_pl[tr] if y_train is None else y_train
    w = prep.w_pl[tr] if w_train is None else w_train
    va, vb, te = np.flatnonzero(prep.val_a), np.flatnonzero(prep.val_b), np.flatnonzero(prep.test)
    Xs = prep.Xs
    y_val = prep.y_true if truth_for_validation else prep.y_pl
    model = train_twxgb(Xs[tr], y, w, Xs[va], y_val[va], params, seed=seed, nthread=n_jobs,
                        es_max_rows=int(cfg.model.es_max_rows))
    thr, f1_cal = calibrate_threshold(model, Xs[vb], y_val[vb])
    score = model.predict_proba(Xs[te])
    m = binary_metrics(prep.y_true[te], score, thr)
    out = {"seed": int(seed), **m, "threshold": float(thr), "n_trees": int(model.n_trees),
           "fit_seconds": round(model.fit_seconds, 2), "val_b_pseudo_f1": float(f1_cal),
           "pseudo_label_test": {k: v for k, v in binary_metrics(prep.y_pl[te], score, thr).items()
                                 if k in ("precision", "recall", "f1", "accuracy")},
           "families": per_family_recall(prep.y_true[te], score > thr, prep.family[te]),
           "objective": model.objective_summary, "train_rows": int(len(tr)), "n_features": int(Xs.shape[1])}
    if keep_scores:
        out["_score"], out["_model"], out["_rows"] = score, model, te
    return out


def strip_private(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def summarise(runs: list[dict], extra_keys: list[str] | None = None) -> dict:
    keys = ["precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc"] + list(extra_keys or [])
    s = summarise_runs(runs, keys)
    fam = {}
    for r in runs:
        for f in r.get("families", []):
            fam.setdefault(f["family"], {"support": f["support"], "recall": []})["recall"].append(f["recall"])
    s["families"] = {k: {"support": v["support"], **seed_summary(v["recall"])} for k, v in fam.items()}
    if fam:
        s["macro_recall"] = seed_summary([float(np.mean([f["recall"] for f in r["families"]])) for r in runs
                                          if r.get("families")])
    return s


def closest_to_mean(runs: list[dict], key: str = "f1") -> dict:
    mu = float(np.mean([r[key] for r in runs]))
    return min(runs, key=lambda r: abs(r[key] - mu))


def run_seeds(prep, params: dict, seeds, cfg, n_jobs: int, **kw) -> tuple[list[dict], dict]:
    runs = []
    for s in seeds:
        t0 = time.perf_counter()
        r = run_detector(prep, params, int(s), cfg, n_jobs, **kw)
        runs.append(r)
        log.info("seed %s: P %.4f R %.4f F1 %.4f (%.0f s)", s, r["precision"], r["recall"], r["f1"],
                 time.perf_counter() - t0)
    return runs, summarise(runs)


def with_bootstrap(run: dict, cfg, seed: int | None = None) -> dict:
    seed = int(cfg.get_path("experiments.bootstrap_seed", 0)) if seed is None else seed
    return bootstrap_counts_ci(run["tp"], run["fp"], run["fn"], run["tn"], B=int(cfg.experiments.bootstrap), seed=seed)


def header(ctx: Ctx, name: str) -> dict:
    return {"experiment": name, "config": ctx.cfg.get("_config_path"), "environment": environment_record(),
            "synthetic": bool(ctx.cfg.get_path("synthetic.enabled", False))}

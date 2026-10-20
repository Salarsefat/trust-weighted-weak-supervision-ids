"""Step 9: hyper-parameter search - randomised start followed by the Tree-structured Parzen Estimator.

The search never sees ground truth: its objective is the pseudo-label F1 on validation rows.
The same routine (same number of trials, same split of random and TPE trials) is used for the
proposed detector and for every baseline.
"""
from __future__ import annotations

import time

import numpy as np
import optuna

from .metrics import best_f1_threshold
from .model import DEFAULT_PARAMS, train_twxgb
from .utils import get_logger

log = get_logger("twws.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "loguniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            v = trial.suggest_categorical(name, [str(c) if c is None else c for c in spec["choices"]])
            out[name] = None if (isinstance(v, str) and v.lower() == "none") else v
        else:  # pragma: no cover
            raise ValueError(f"unknown search-space type '{kind}'")
    return out


def run_search(objective, space: dict, n_trials: int, n_random: int, seed: int) -> dict:
    """Maximise ``objective(params) -> float``; returns best parameters and the trial log."""
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=int(n_random))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.perf_counter()
    rows = []

    def _obj(trial):
        params = suggest(trial, space)
        t = time.perf_counter()
        val = float(objective(params))
        rows.append({"trial": trial.number, "value": val, "seconds": round(time.perf_counter() - t, 2),
                     "phase": "random" if trial.number < n_random else "tpe", **params})
        return val

    study.optimize(_obj, n_trials=int(n_trials))
    best = suggest_fixed(study.best_params)
    return {"best_params": best, "best_value": float(study.best_value), "n_trials": int(n_trials),
            "n_random": int(n_random), "sampler": "TPESampler", "seed": int(seed),
            "seconds": round(time.perf_counter() - t0, 1), "trials": rows}


def suggest_fixed(params: dict) -> dict:
    return {k: (None if (isinstance(v, str) and v.lower() == "none") else v) for k, v in params.items()}


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a subsample of size <= n that keeps the class proportions of ``y``."""
    y = np.asarray(y)
    if len(y) <= n:
        return np.arange(len(y))
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, int(round(n * len(idx) / len(y))))
        out.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def search_subsets(prep, cfg, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (into the corpus) used inside the search: training part and validation part."""
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(prep.usable_train())
    va = np.flatnonzero(prep.val_a)
    tr = tr[stratified_subsample(prep.y_pl[tr], int(cfg.search.search_rows), rng)]
    va = va[stratified_subsample(prep.y_pl[va], int(cfg.search.val_rows), rng)]
    return tr, va


def search_twxgb(prep, cfg, nthread: int | None = None) -> dict:
    """Search the tree and objective hyper-parameters of the proposed detector."""
    tr, va = search_subsets(prep, cfg, int(cfg.search.seed))
    Xtr, ytr, wtr = prep.Xs[tr], prep.y_pl[tr], prep.w_pl[tr]
    Xva, yva = prep.Xs[va], prep.y_pl[va]
    base = {**DEFAULT_PARAMS, **dict(cfg.model.defaults), "num_boost_round": int(cfg.search.num_boost_round),
            "early_stopping_rounds": int(cfg.search.early_stopping_rounds)}

    def objective(params: dict) -> float:
        m = train_twxgb(Xtr, ytr, wtr, Xva, yva, {**base, **params}, seed=int(cfg.search.seed), nthread=nthread,
                        es_max_rows=int(cfg.model.es_max_rows))
        return best_f1_threshold(yva, m.predict_proba(Xva))[1]

    res = run_search(objective, {k: dict(v) for k, v in dict(cfg.search.space).items()}, int(cfg.search.n_trials),
                     int(cfg.search.n_random), int(cfg.search.seed))
    res["search_rows"] = int(len(tr))
    res["val_rows"] = int(len(va))
    res["criterion"] = "pseudo-label F1 on validation rows (threshold maximising F1)"
    res["space"] = {k: dict(v) for k, v in dict(cfg.search.space).items()}
    return res

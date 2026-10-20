"""Training of the trust-weighted XGBoost detector, early stopping and threshold calibration."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from .metrics import best_f1_threshold
from .objective import TrustFocalObjective, sigmoid
from .utils import get_logger

log = get_logger("twws.model")

DEFAULT_PARAMS = {
    # tree ensemble
    "eta": 0.05, "max_depth": 6, "gamma_split": 1.0, "reg_lambda": 1.0, "min_child_weight": 1.0,
    "colsample_bytree": 0.8, "subsample": 0.8, "num_boost_round": 800, "early_stopping_rounds": 50, "max_bin": 256,
    # objective
    "focal_gamma": 2.0, "delta": 0.1, "class_weight": "reciprocal", "hess_floor": 1e-6, "warmup_rounds": 20,
    "warmup_cap": 0.5, "use_trust": True,
}


@dataclass
class TWModel:
    booster: xgb.Booster
    best_iteration: int
    params: dict
    objective_summary: dict
    fit_seconds: float
    es_history: list = field(default_factory=list)
    nthread: int = 0

    def margin(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X, nthread=self.nthread or -1)
        return self.booster.predict(d, output_margin=True, iteration_range=(0, self.best_iteration + 1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.margin(X)).astype(np.float32)

    @property
    def n_trees(self) -> int:
        return self.best_iteration + 1


def _xgb_params(p: dict, seed: int, nthread: int) -> dict:
    out = {"eta": p["eta"], "max_depth": int(p["max_depth"]), "gamma": p["gamma_split"], "lambda": p["reg_lambda"],
           "min_child_weight": p["min_child_weight"], "colsample_bytree": p["colsample_bytree"],
           "subsample": 1.0,            # rows are drawn inside the objective so that alpha tracks the round
           "tree_method": "hist", "max_bin": int(p.get("max_bin", 256)), "base_score": 0.0, "seed": int(seed),
           "nthread": int(nthread), "verbosity": 0}
    if p.get("device"):
        out["device"] = p["device"]
    return out


def train_twxgb(X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray | None, X_es: np.ndarray,
                y_es: np.ndarray, params: dict | None = None, seed: int = 0, nthread: int | None = None,
                es_max_rows: int = 100_000) -> TWModel:
    """Fit the detector on pseudo-labels; early stopping on pseudo-label PR-AUC of the ES rows."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    nthread = nthread or os.cpu_count() or 1
    rng = np.random.default_rng(seed + 7919)
    if len(y_es) > es_max_rows:
        sub = rng.choice(len(y_es), size=es_max_rows, replace=False)
        X_es, y_es = X_es[sub], np.asarray(y_es)[sub]
    dtrain = xgb.DMatrix(X_train, label=np.asarray(y_train, dtype=np.float32), nthread=nthread)
    des = xgb.DMatrix(X_es, label=np.asarray(y_es, dtype=np.float32), nthread=nthread)
    obj = TrustFocalObjective(y_train, w_train, gamma=p["focal_gamma"], class_weight=p["class_weight"],
                              delta=p["delta"], hess_floor=p["hess_floor"], subsample=p["subsample"],
                              warmup_rounds=p["warmup_rounds"], warmup_cap=p["warmup_cap"],
                              use_trust=p["use_trust"], seed=seed)
    y_es_i = np.asarray(y_es).astype(int)
    two_classes = 0 < y_es_i.sum() < len(y_es_i)

    def metric(predt, dmat):
        if not two_classes:
            return "pl_prauc", 0.0
        return "pl_prauc", float(average_precision_score(y_es_i, predt))

    hist: dict = {}
    t0 = time.perf_counter()
    booster = xgb.train(_xgb_params(p, seed, nthread), dtrain, num_boost_round=int(p["num_boost_round"]), obj=obj,
                        custom_metric=metric, evals=[(des, "es")], maximize=True,
                        early_stopping_rounds=int(p["early_stopping_rounds"]) if two_classes else None,
                        evals_result=hist, verbose_eval=False)
    dt = time.perf_counter() - t0
    best = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return TWModel(booster, best, p, obj.summary(), dt, hist.get("es", {}).get("pl_prauc", []), nthread)


def calibrate_threshold(model, X_cal: np.ndarray, y_cal: np.ndarray) -> tuple[float, float]:
    """Decision threshold maximising pseudo-label F1 on the calibration rows."""
    return best_f1_threshold(y_cal, model.predict_proba(X_cal))

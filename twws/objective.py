"""Steps 7-8: trust-weighted focal objective for XGBoost.

For raw score z and p = sigmoid(z) the focal loss is

    l(y, p) = -(1 - p)^gamma * y * log(p) - p^gamma * (1 - y) * log(1 - p).

Its first and second derivatives with respect to z are (Eqs. 18-21 of the manuscript)

    y = 1:  g = (1-p)^gamma [gamma p ln p - (1-p)]
            h = p (1-p)^gamma [-gamma^2 p ln p + gamma (1-p) ln p + (2 gamma + 1)(1-p)]
    y = 0:  g = p^gamma [p - gamma (1-p) ln(1-p)]
            h = p^gamma (1-p) [-gamma^2 (1-p) ln(1-p) + gamma p ln(1-p) + (2 gamma + 1) p]

(gamma = 0 gives the logistic gradient p - y and Hessian p (1-p)).  The Hessian is negative
for confidently misclassified rows, so it is floored at ``hess_floor``.

Per round t the derivatives are re-weighted as

    g_i <- clip(tau_i * alpha_i^(t) * g_i, -delta, +delta),   h_i <- max(tau_i * alpha_i^(t) * h_i, hess_floor)

with trust tau_i and class weight alpha_i^(t) computed on the rows drawn for round t.
"""
from __future__ import annotations

import numpy as np

CLASS_WEIGHT_MODES = ("reciprocal", "balanced", "none")


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def focal_grad_hess(y: np.ndarray, p: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and (unfloored) Hessian of the focal loss with respect to the raw score."""
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    q = 1.0 - p
    lp, lq = np.log(p), np.log(q)
    if gamma == 0:
        g1, h1 = -q, p * q
        g0, h0 = p, p * q
    else:
        g1 = q ** gamma * (gamma * p * lp - q)
        h1 = p * q ** gamma * (-gamma ** 2 * p * lp + gamma * q * lp + (2 * gamma + 1) * q)
        g0 = p ** gamma * (p - gamma * q * lq)
        h0 = p ** gamma * q * (-gamma ** 2 * q * lq + gamma * p * lq + (2 * gamma + 1) * p)
    pos = np.asarray(y) == 1
    return np.where(pos, g1, g0), np.where(pos, h1, h0)


def focal_loss(y: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.where(np.asarray(y) == 1, -((1 - p) ** gamma) * np.log(p), -(p ** gamma) * np.log(1 - p))


def class_weights(n0: float, n1: float, mode: str) -> tuple[float, float]:
    """(alpha for benign rows, alpha for malicious rows) from the class counts of one round.

    reciprocal  alpha_c = (N_other + 1) / (N_c + 1)          -- Eq. (23) of the first submission
    balanced    alpha_c = (N_0 + N_1 + 2) / (2 (N_c + 1))    -- equal total weight per class
    none        alpha_c = 1
    """
    mode = "none" if mode is None else str(mode).lower()
    if mode == "reciprocal":
        return (n1 + 1.0) / (n0 + 1.0), (n0 + 1.0) / (n1 + 1.0)
    if mode == "balanced":
        tot = n0 + n1 + 2.0
        return tot / (2.0 * (n0 + 1.0)), tot / (2.0 * (n1 + 1.0))
    if mode == "none":
        return 1.0, 1.0
    raise ValueError(f"unknown class-weight mode '{mode}'")


class TrustFocalObjective:
    """Callable custom objective for ``xgboost.train``.

    Parameters
    ----------
    y, trust       pseudo-labels (0/1) and per-row trust weights of the training matrix
    gamma          focal focusing parameter (0 = logistic loss)
    class_weight   'reciprocal', 'balanced' or 'none'
    delta          gradient clip applied *after* trust and class weighting (None = no clipping)
    hess_floor     lower bound of the Hessian
    subsample      share of rows drawn (Bernoulli) for each boosting round; the class counts
                   behind alpha are taken on the drawn rows, undrawn rows get g = h = 0
    warmup_rounds  during the first rounds the raw gradient is additionally capped at
                   ``warmup_cap * delta`` before weighting (hard-instance capping)
    use_trust      False replaces every trust weight by 1 (ablation)
    """

    def __init__(self, y: np.ndarray, trust: np.ndarray | None, gamma: float = 2.0, class_weight: str = "reciprocal",
                 delta: float | None = 0.1, hess_floor: float = 1e-6, subsample: float = 0.8, warmup_rounds: int = 20,
                 warmup_cap: float = 0.5, use_trust: bool = True, seed: int = 0) -> None:
        self.y = np.asarray(y).astype(np.int8)
        self.pos = self.y == 1
        t = np.ones(len(self.y)) if (trust is None or not use_trust) else np.asarray(trust, dtype=np.float64)
        self.trust = t
        self.gamma = float(gamma)
        self.class_weight = "none" if class_weight is None else str(class_weight).lower()
        class_weights(1.0, 1.0, self.class_weight)        # fail at construction, not in the first boosting round
        self.delta = None if delta in (None, 0, "none") else float(delta)
        self.hess_floor, self.subsample = float(hess_floor), float(subsample)
        self.warmup_rounds, self.warmup_cap = int(warmup_rounds), float(warmup_cap)
        self.rng = np.random.default_rng(seed)
        self.round = 0
        self.alpha_history: list[tuple[float, float]] = []
        self.clip_share_history: list[float] = []
        self.neg_hess_share_history: list[float] = []

    def __call__(self, predt: np.ndarray, dtrain=None):
        z = np.asarray(predt, dtype=np.float64).reshape(-1)
        p = sigmoid(z)
        g, h = focal_grad_hess(self.y, p, self.gamma)
        self.neg_hess_share_history.append(float((h < 0).mean()))
        h = np.maximum(h, self.hess_floor)
        n = len(z)
        if self.subsample < 1.0:
            mask = self.rng.random(n) < self.subsample
        else:
            mask = np.ones(n, dtype=bool)
        n1 = float((self.pos & mask).sum())
        n0 = float(mask.sum()) - n1
        a0, a1 = class_weights(n0, n1, self.class_weight)
        self.alpha_history.append((a0, a1))
        if self.delta is not None and self.round < self.warmup_rounds:
            cap = self.warmup_cap * self.delta
            g = np.clip(g, -cap, cap)
        w = self.trust * np.where(self.pos, a1, a0)
        g = w * g
        if self.delta is not None:
            clipped = np.abs(g) > self.delta
            self.clip_share_history.append(float(clipped[mask].mean()) if mask.any() else 0.0)
            g = np.clip(g, -self.delta, self.delta)
        else:
            self.clip_share_history.append(0.0)
        h = np.maximum(w * h, self.hess_floor)
        g = np.where(mask, g, 0.0)
        h = np.where(mask, h, 0.0)
        self.round += 1
        return g, h

    def summary(self) -> dict:
        a = np.array(self.alpha_history) if self.alpha_history else np.zeros((0, 2))
        return {
            "gamma": self.gamma, "class_weight": self.class_weight, "delta": self.delta,
            "hess_floor": self.hess_floor, "subsample": self.subsample, "warmup_rounds": self.warmup_rounds,
            "rounds_called": self.round,
            "alpha_benign_mean": float(a[:, 0].mean()) if len(a) else None,
            "alpha_malicious_mean": float(a[:, 1].mean()) if len(a) else None,
            "alpha_malicious_rel_range": float((a[:, 1].max() - a[:, 1].min()) / a[:, 1].mean()) if len(a) else None,
            "clipped_share_mean": float(np.mean(self.clip_share_history)) if self.clip_share_history else None,
            "clipped_share_last": float(self.clip_share_history[-1]) if self.clip_share_history else None,
            "negative_hessian_share_max": float(np.max(self.neg_hess_share_history)) if self.neg_hess_share_history else None,
        }

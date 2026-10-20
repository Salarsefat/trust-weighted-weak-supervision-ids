"""Evaluation metrics, threshold selection and uncertainty estimates."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

METRIC_KEYS = ["precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc"]


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def from_counts(tp: float, fp: float, fn: float, tn: float) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / max(1.0, tp + fp + fn + tn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "fpr": fpr}


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Threshold metrics plus ranking metrics; ``y`` is 0/1 ground truth (or pseudo-labels)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    c = confusion(y, score > threshold)
    out = {**from_counts(**c), **c, "threshold": float(threshold), "n": int(len(y)), "positives": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auc_roc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc_roc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on (y, score); returns (threshold, F1). Predictions are score > threshold."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    # precision_recall_curve uses score >= thr; move just below so that "score > t" is equivalent
    return float(np.nextafter(thr[i], -np.inf)), float(f1[i])


def per_family_recall(y: np.ndarray, pred: np.ndarray, family: np.ndarray) -> list[dict]:
    """Support, TP, FN and recall for each attack family (defined for a binary detector)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    fam = np.asarray(family).astype(str)
    rows = []
    for f in sorted(set(fam[y])):
        m = y & (fam == f)
        tp = int((m & pred).sum())
        n = int(m.sum())
        rows.append({"family": f, "support": n, "tp": tp, "fn": n - tp, "recall": tp / n if n else float("nan")})
    return rows


def seed_summary(values: list[float], level: float = 0.95) -> dict:
    """Mean, sample standard deviation and t-based confidence half-width across seeds."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci": float("nan"), "n": 0}
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    ci = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "std": sd, "ci": ci, "n": n, "min": float(v.min()), "max": float(v.max())}


def summarise_runs(runs: list[dict], keys: list[str] | None = None) -> dict:
    keys = keys or METRIC_KEYS
    return {k: seed_summary([r.get(k) for r in runs]) for k in keys}


def bootstrap_counts_ci(tp: int, fp: int, fn: int, tn: int, B: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap over test flows for threshold metrics.

    Resampling flows with replacement is equivalent to drawing the four confusion-matrix
    cells from a multinomial with the observed proportions.
    """
    rng = np.random.default_rng(seed)
    n = tp + fp + fn + tn
    draws = rng.multinomial(n, np.array([tp, fp, fn, tn]) / n, size=B).astype(np.float64)
    a, b, c, d = draws.T
    with np.errstate(divide="ignore", invalid="ignore"):
        m = {"precision": a / (a + b), "recall": a / (a + c), "f1": 2 * a / (2 * a + b + c),
             "accuracy": (a + d) / n, "fpr": b / (b + d)}
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {k: {"lo": float(np.nanquantile(v, lo)), "hi": float(np.nanquantile(v, hi))} for k, v in m.items()}


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval of a binomial proportion k/n.

    Used for the recall of one attack family over its test flows, where the bootstrap over all test flows
    is not reported and a family may have only a handful of flows (0 of 3 detected is compatible with a
    recall of up to 0.71).  Returns (nan, nan) for n = 0.
    """
    k, n = int(k), int(n)
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def paired_bootstrap_f1(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the F1 difference (a - b) of two detectors evaluated on the same test flows."""
    y = np.asarray(y).astype(int)
    a = np.asarray(pred_a).astype(int)
    b = np.asarray(pred_b).astype(int)
    code = y * 4 + a * 2 + b                              # 8 joint outcomes
    counts = np.bincount(code, minlength=8).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(y), counts / counts.sum(), size=B).astype(np.float64)

    def f1(tp, fp, fn):
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    tp_a = draws[:, 6] + draws[:, 7]
    fn_a = draws[:, 4] + draws[:, 5]
    fp_a = draws[:, 2] + draws[:, 3]
    tp_b = draws[:, 5] + draws[:, 7]
    fn_b = draws[:, 4] + draws[:, 6]
    fp_b = draws[:, 1] + draws[:, 3]
    diff = f1(tp_a, fp_a, fn_a) - f1(tp_b, fp_b, fn_b)
    return {"diff_mean": float(diff.mean()), "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975)),
            "p_two_sided": float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))}

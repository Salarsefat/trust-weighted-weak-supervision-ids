import numpy as np

from twws.metrics import (best_f1_threshold, binary_metrics, bootstrap_counts_ci, clopper_pearson, confusion,
                          paired_bootstrap_f1, per_family_recall, seed_summary)


def test_metrics_and_family_table_add_up():
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(0.6 * y + rng.normal(0.2, 0.2, n), 0, 1)
    m = binary_metrics(y, score, 0.5)
    c = confusion(y, score > 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (c["tp"], c["fp"], c["fn"], c["tn"])
    assert m["tp"] + m["fn"] == y.sum() and m["fp"] + m["tn"] == n - y.sum()
    fam = np.where(y == 1, rng.choice(["A", "B", "C"], n), "Benign")
    rows = per_family_recall(y, score > 0.5, fam)
    assert sum(r["tp"] for r in rows) == m["tp"] and sum(r["fn"] for r in rows) == m["fn"]
    thr, f1 = best_f1_threshold(y, score)
    assert f1 >= m["f1"] - 1e-9


def test_intervals():
    s = seed_summary([0.90, 0.91, 0.92, 0.93, 0.94])
    assert abs(s["mean"] - 0.92) < 1e-12 and s["ci"] > 0 and s["n"] == 5
    ci = bootstrap_counts_ci(800, 50, 120, 9000, B=500, seed=0)
    assert ci["f1"]["lo"] < 2 * 800 / (2 * 800 + 50 + 120) < ci["f1"]["hi"]
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.3).astype(int)
    a = np.where(rng.random(4000) < 0.95, y, 1 - y).astype(bool)
    b = np.where(rng.random(4000) < 0.80, y, 1 - y).astype(bool)
    d = paired_bootstrap_f1(y, a, b, B=400, seed=0)
    assert d["lo"] > 0 and d["diff_mean"] > 0


def test_clopper_pearson():
    # zero successes: the upper end is 1 - (alpha/2)^(1/n)
    lo, hi = clopper_pearson(0, 3)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 3))) < 1e-9          # 0.708
    lo, hi = clopper_pearson(0, 147)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 147))) < 1e-9        # 0.025
    lo, hi = clopper_pearson(5, 5)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 5)) < 1e-9
    # symmetric case and coverage of the point estimate
    lo, hi = clopper_pearson(4231, 8462)
    assert lo < 0.5 < hi and abs((0.5 - lo) - (hi - 0.5)) < 1e-6 and hi - lo < 0.025
    assert all(map(lambda x: x != x, clopper_pearson(0, 0)))              # nan for an empty family

"""Vote aggregation, trust and refinement."""
import numpy as np

from twws.weak import NO_MAJORITY, aggregate_votes, dawid_skene, high_trust_mask, majority_vote, refine


def test_majority_vote_semantics():
    V = np.array([[1, 1, 1, 0, -1],     # 3 malicious, 1 benign  -> malicious, trust 3/5
                  [0, 0, -1, -1, -1],   # 2 benign               -> benign,    trust 2/5
                  [1, 0, -1, -1, -1],   # tie                    -> no majority
                  [-1, -1, -1, -1, -1]])  # all abstain            -> no majority
    out = majority_vote(V)
    assert out["label"].tolist() == [1, 0, NO_MAJORITY, NO_MAJORITY]
    assert np.allclose(out["trust"], [0.6, 0.4, 0.0, 0.0])


def _simulate(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.17).astype(int)
    sens = np.array([0.05, 0.50, 0.02, 0.85, 0.95])
    fa = np.array([0.005, 0.004, 0.001, 0.02, 0.0005])
    cover = np.array([0.99, 1.0, 0.99, 0.97, 0.4])
    V = np.full((n, 5), -1, dtype=np.int8)
    for k in range(5):
        voted = rng.random(n) < cover[k]
        p = np.where(y == 1, sens[k], fa[k])
        V[voted, k] = (rng.random(voted.sum()) < p[voted]).astype(np.int8)
    return V, y, sens, fa


def test_label_model_recovers_planted_error_rates_without_labels():
    V, y, sens, fa = _simulate()
    ds = dawid_skene(V)
    assert abs(ds["pi"] - y.mean()) < 0.02
    assert np.max(np.abs(np.array(ds["sensitivity"]) - sens)) < 0.05
    assert np.max(np.abs(np.array(ds["false_alarm"]) - fa)) < 0.01
    agg = aggregate_votes(V, "ds")
    assert (agg["label"] == y).mean() > 0.95
    assert ((agg["trust"] >= 0.5) & (agg["trust"] <= 1.0)).all()


def test_equal_vote_majority_cannot_reach_high_trust_for_rare_positive_votes():
    """Three of five oracles vote 'benign' by default, so a majority for 'malicious' is rare."""
    V, y, _, _ = _simulate()
    maj = aggregate_votes(V, "majority")
    ds = aggregate_votes(V, "ds")
    rec_maj = ((maj["label"] == 1) & (y == 1)).sum() / y.sum()
    rec_ds = ((ds["label"] == 1) & (y == 1)).sum() / y.sum()
    assert rec_maj < 0.2 < rec_ds


def test_label_switching_is_excluded():
    # an oracle that votes malicious more often on benign flows must not flip the classes
    rng = np.random.default_rng(1)
    n = 20000
    y = (rng.random(n) < 0.2).astype(int)
    good = np.where(y == 1, rng.random(n) < 0.9, rng.random(n) < 0.01).astype(np.int8)
    bad = np.where(y == 1, rng.random(n) < 0.05, rng.random(n) < 0.3).astype(np.int8)
    good2 = np.where(y == 1, rng.random(n) < 0.8, rng.random(n) < 0.02).astype(np.int8)
    ds = dawid_skene(np.column_stack([good, bad, good2]))
    assert all(a >= b - 1e-12 for a, b in zip(ds["sensitivity"], ds["false_alarm"]))
    q = ds["q"]
    assert ((q >= 0.5).astype(int) == y).mean() > 0.9


def test_refinement_relabels_only_low_trust_rows():
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.3).astype(np.int8)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[:, 0] += 4.0 * y
    label = y.copy()
    trust = np.full(n, 0.9, dtype=np.float32)
    low = rng.random(n) < 0.1
    trust[low] = 0.55
    label[low] = 1 - label[low]                       # low-trust rows carry the wrong label
    agg = {"label": label, "trust": trust}
    out = refine(X, agg, fit_mask=np.ones(n, bool), cfg={"high_trust": 0.8, "p_flip": 0.85, "refined_trust": 0.6})
    assert out["refined"].sum() == low.sum()
    assert (out["label"][~low] == label[~low]).all()            # high-trust rows untouched
    assert np.allclose(out["weight"][low], 0.6) and np.allclose(out["weight"][~low], 0.9)
    assert (out["label"][low] == y[low]).mean() > 0.9           # most wrong labels are repaired
    assert high_trust_mask(agg, 0.8).sum() == (~low).sum()


def test_majority_trust_table_bins_every_flow_by_agreeing_oracles():
    """The trust table of equal-vote majority has one row per number k of agreeing oracles (k/K for K oracles);
    with seven oracles the five-oracle levels 0.2, ..., 1.0 of earlier versions would miss most flows."""
    from twws.experiments.core import _trust_table
    from twws.weak import aggregate_votes
    rng = np.random.default_rng(0)
    K, n = 7, 4000
    y = (rng.random(n) < 0.3).astype(np.int8)
    V = np.where(rng.random((n, K)) < 0.8, y[:, None], 1 - y[:, None]).astype(np.int8)
    V[rng.random((n, K)) < 0.3] = -1                                   # abstentions
    a = aggregate_votes(V, "majority")
    rows = _trust_table(a, y, "majority", n_oracles=K)
    assert [r["trust"] for r in rows][:2] == ["no majority", "1/7"] and rows[-1]["trust"] == "7/7"
    assert sum(r["n"] for r in rows) == n and not any(r["trust"] == "not binned" for r in rows)
    agree = np.rint(a["trust"] * K).astype(int)
    for k in range(1, K + 1):
        r = next(r for r in rows if r["trust"] == f"{k}/{K}")
        assert r["n"] == int(((a["label"] != -1) & (agree == k)).sum())
    ds_rows = _trust_table(aggregate_votes(V, "ds", fit_mask=np.ones(n, bool)), y, "ds")
    assert sum(r["n"] for r in ds_rows) == n

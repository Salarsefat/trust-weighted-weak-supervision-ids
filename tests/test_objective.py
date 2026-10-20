"""The custom objective: derivatives, weighting order, clipping, sub-sampling."""
import numpy as np
import pytest

from twws.objective import TrustFocalObjective, class_weights, focal_grad_hess, focal_loss, sigmoid


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_gradient_and_hessian_match_finite_differences(gamma):
    z = np.linspace(-6, 6, 241)
    eps = 1e-4
    for y in (0, 1):
        yy = np.full_like(z, y)
        g, h = focal_grad_hess(yy, sigmoid(z), gamma)
        L = lambda t: focal_loss(yy, sigmoid(t), gamma)
        g_num = (L(z + eps) - L(z - eps)) / (2 * eps)
        assert np.max(np.abs(g - g_num)) < 1e-6                      # gradient against the loss
        G = lambda t: focal_grad_hess(yy, sigmoid(t), gamma)[0]
        h_num = (G(z + eps) - G(z - eps)) / (2 * eps)
        assert np.max(np.abs(h - h_num)) < 1e-6                      # Hessian against the gradient


def test_gamma_zero_is_the_logistic_loss():
    z = np.linspace(-5, 5, 101)
    p = sigmoid(z)
    for y in (0, 1):
        g, h = focal_grad_hess(np.full_like(z, y), p, 0.0)
        assert np.allclose(g, p - y)
        assert np.allclose(h, p * (1 - p))


def test_focal_hessian_changes_sign_for_confident_errors():
    # gamma = 2, malicious row: the Hessian is negative for small p and positive for large p
    p = np.array([0.01, 0.03, 0.05, 0.07, 0.2, 0.9])
    _, h = focal_grad_hess(np.ones_like(p), p, 2.0)
    assert (h[:3] < 0).all() and (h[3:] > 0).all()


def test_class_weight_modes():
    a0, a1 = class_weights(830, 170, "reciprocal")
    assert a1 == pytest.approx(831 / 171) and a0 == pytest.approx(171 / 831)
    b0, b1 = class_weights(830, 170, "balanced")
    assert b0 * 831 == pytest.approx(b1 * 171)          # equal total weight per class
    assert class_weights(5, 7, "none") == (1.0, 1.0)


def test_clipping_is_applied_after_weighting_and_undrawn_rows_are_zero():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    trust = rng.uniform(0.5, 1.0, n)
    obj = TrustFocalObjective(y, trust, gamma=2.0, class_weight="reciprocal", delta=0.1, subsample=0.7,
                              warmup_rounds=0, seed=3)
    g, h = obj(np.zeros(n))
    assert np.abs(g).max() <= 0.1 + 1e-12                # the bound holds for the weighted gradient
    drawn = h > 0
    assert 0.6 < drawn.mean() < 0.8
    assert np.all(g[~drawn] == 0) and np.all(h[~drawn] == 0)
    assert (h[drawn] >= obj.hess_floor).all()


def test_trust_scales_unclipped_gradients():
    y = np.array([1, 1, 0, 0])
    z = np.zeros(4)
    lo = TrustFocalObjective(y, np.array([0.5, 0.5, 0.5, 0.5]), gamma=0.0, class_weight="none", delta=None, subsample=1.0)
    hi = TrustFocalObjective(y, np.ones(4), gamma=0.0, class_weight="none", delta=None, subsample=1.0)
    g_lo, h_lo = lo(z)
    g_hi, h_hi = hi(z)
    assert np.allclose(g_lo, 0.5 * g_hi) and np.allclose(h_lo, 0.5 * h_hi)
    off = TrustFocalObjective(y, np.array([0.5, 0.5, 0.5, 0.5]), gamma=0.0, class_weight="none", delta=None,
                              subsample=1.0, use_trust=False)
    assert np.allclose(off(z)[0], g_hi)


def test_detector_learns_a_separable_problem():
    from twws.metrics import binary_metrics
    from twws.model import calibrate_threshold, train_twxgb

    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) < 0.2).astype(int)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] += 2.5 * y
    w = np.ones(n, dtype=np.float32)
    m = train_twxgb(X[:4000], y[:4000], w[:4000], X[4000:5000], y[4000:5000],
                    {"num_boost_round": 80, "early_stopping_rounds": 15}, seed=0, nthread=2)
    thr, _ = calibrate_threshold(m, X[4000:5000], y[4000:5000])
    met = binary_metrics(y[5000:], m.predict_proba(X[5000:]), thr)
    assert met["f1"] > 0.75 and met["auc_roc"] > 0.93


def test_class_weight_none_from_the_search_space():
    """The search space spells the unweighted option 'none'; the sampler hands it over as None.
    Both spellings must train (regression test: None used to raise in the first boosting round)."""
    import numpy as np
    import pytest

    from twws.model import train_twxgb
    from twws.objective import TrustFocalObjective, class_weights
    from twws.tuning import suggest_fixed

    assert class_weights(10, 5, None) == (1.0, 1.0) == class_weights(10, 5, "none")
    assert suggest_fixed({"class_weight": "none", "delta": "none"}) == {"class_weight": None, "delta": None}
    with pytest.raises(ValueError):
        TrustFocalObjective(np.array([0, 1]), None, class_weight="inverse")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0.8).astype(int)
    for cw in (None, "none", "balanced", "reciprocal"):
        for delta in (None, 0.3):
            fit = train_twxgb(X[:300], y[:300], np.ones(300), X[300:], y[300:],
                              {"class_weight": cw, "delta": delta, "num_boost_round": 5, "early_stopping_rounds": 5,
                               "max_depth": 2}, seed=1, nthread=1)
            assert fit is not None

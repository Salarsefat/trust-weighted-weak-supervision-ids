"""Step 2 (aggregation) and Step 5(4): vote aggregation, trust scores and semi-supervised refinement."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .utils import get_logger

log = get_logger("twws.weak")

NO_MAJORITY = -1


def majority_vote(votes: np.ndarray) -> dict:
    """Equal-vote majority over non-abstaining oracles and the agreement-fraction trust score.

    votes   (n, K) array with entries 1 (malicious), 0 (benign), -1 (abstain)

    Returns ``label`` (1, 0, or -1 when the non-abstaining votes are tied or every oracle
    abstains) and ``trust`` = (number of oracles voting for the label) / K (0 without a
    majority).  This is the aggregation rule of the first submission.
    """
    votes = np.asarray(votes)
    K = votes.shape[1]
    n_mal = (votes == 1).sum(axis=1)
    n_ben = (votes == 0).sum(axis=1)
    label = np.where(n_mal > n_ben, 1, np.where(n_ben > n_mal, 0, NO_MAJORITY)).astype(np.int8)
    agree = np.where(label == 1, n_mal, np.where(label == 0, n_ben, 0))
    return {"label": label, "trust": (agree / K).astype(np.float32)}


def dawid_skene(votes: np.ndarray, n_iter: int = 500, tol: float = 1e-9, prior_mal: float = 0.1,
                init_sens: float = 0.7, init_fa: float = 0.05, pseudo: float = 1.0,
                fit_mask: np.ndarray | None = None, q0: np.ndarray | None = None) -> dict:
    """Two-class Dawid-Skene label model fitted by EM without any ground truth.

    Each oracle k is described by its sensitivity ``a_k = P(vote malicious | malicious)`` and
    false-alarm rate ``b_k = P(vote malicious | benign)`` among the flows on which it votes;
    abstentions are treated as missing.  ``pi`` is the share of malicious flows.  The model
    assumes that oracles err independently given the true class.  Label switching is
    excluded by the constraint ``a_k >= b_k`` (a malicious vote never counts as evidence for
    benign); an oracle violating it becomes uninformative (a_k = b_k).

    ``fit_mask`` restricts the estimation to some rows (the training split); the posterior is
    then computed for every row with the fitted parameters, so that votes of validation and
    test flows never influence the model.

    ``q0`` (optional, one value in [0, 1] per row) replaces the default starting point: EM then
    starts with an M-step that treats ``q0`` as the posterior of the fitted rows.

    Returns the posterior ``q = P(malicious | votes)`` per flow and the fitted parameters.
    EM runs on the distinct vote patterns, so the cost does not depend on the number of flows.
    """
    votes = np.asarray(votes).astype(np.int8)
    K = votes.shape[1]
    pat, inv = np.unique(votes, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    fm = np.ones(len(inv), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    rows = inv[fm]
    cnt = np.bincount(rows, minlength=len(pat)).astype(np.float64)
    is_m, is_b = (pat == 1), (pat == 0)
    pi = float(prior_mal)
    a = np.full(K, init_sens)
    b = np.full(K, init_fa)
    if q0 is not None:
        q0 = np.asarray(q0, dtype=np.float64)
        qp = np.bincount(rows, weights=q0[fm], minlength=len(pat)) / np.maximum(cnt, 1.0)
        w1, w0 = cnt * qp, cnt * (1 - qp)
        pi = float(np.clip((w1.sum() + pseudo) / (cnt.sum() + 2 * pseudo), 1e-6, 0.5))
        a = ((w1[:, None] * is_m).sum(axis=0) + pseudo) / ((w1[:, None] * (is_m | is_b)).sum(axis=0) + 2 * pseudo)
        b = ((w0[:, None] * is_m).sum(axis=0) + pseudo) / ((w0[:, None] * (is_m | is_b)).sum(axis=0) + 2 * pseudo)
    q = np.zeros(len(pat))
    ll_old = -np.inf
    it = 0
    for it in range(1, n_iter + 1):
        a_c, b_c = np.clip(a, 1e-6, 1 - 1e-6), np.clip(b, 1e-6, 1 - 1e-6)
        l1 = np.log(pi) + (is_m * np.log(a_c) + is_b * np.log(1 - a_c)).sum(axis=1)
        l0 = np.log(1 - pi) + (is_m * np.log(b_c) + is_b * np.log(1 - b_c)).sum(axis=1)
        mx = np.maximum(l1, l0)
        norm = mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))
        q = np.exp(l1 - norm)
        ll = float((cnt * norm).sum())
        w1, w0 = cnt * q, cnt * (1 - q)
        pi = float(np.clip((w1.sum() + pseudo) / (cnt.sum() + 2 * pseudo), 1e-6, 0.5))
        a = ((w1[:, None] * is_m).sum(axis=0) + pseudo) / ((w1[:, None] * (is_m | is_b)).sum(axis=0) + 2 * pseudo)
        b = ((w0[:, None] * is_m).sum(axis=0) + pseudo) / ((w0[:, None] * (is_m | is_b)).sum(axis=0) + 2 * pseudo)
        bad = a < b
        if bad.any():
            mid = (a[bad] + b[bad]) / 2
            a[bad], b[bad] = mid, mid
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    # posterior of every pattern under the final parameters (also for patterns unseen during fitting)
    a_c, b_c = np.clip(a, 1e-6, 1 - 1e-6), np.clip(b, 1e-6, 1 - 1e-6)
    l1 = np.log(pi) + (is_m * np.log(a_c) + is_b * np.log(1 - a_c)).sum(axis=1)
    l0 = np.log(1 - pi) + (is_m * np.log(b_c) + is_b * np.log(1 - b_c)).sum(axis=1)
    q = 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -700, 700)))
    mx = np.maximum(l1, l0)
    ll_final = float((cnt * (mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx)))).sum())
    return {"q": q[inv].astype(np.float32), "pi": pi, "sensitivity": a.tolist(), "false_alarm": b.tolist(),
            "iterations": it, "log_likelihood": ll_old if np.isfinite(ll_old) else ll, "n_patterns": int(len(pat)),
            "fit_rows": int(cnt.sum()), "log_likelihood_final": ll_final, "converged": bool(it < n_iter)}


def dependent_groups(votes: np.ndarray, fit_mask: np.ndarray | None = None, jaccard_min: float = 0.5) -> dict:
    """Groups of oracles whose malicious votes coincide (docs/ORACLES_EXTENDED.md, change 1).

    Two oracles are linked when the Jaccard index of their malicious votes on the fitted rows is at least
    ``jaccard_min``; groups are the connected components.  Uses no label.
    """
    votes = np.asarray(votes)
    M = votes if fit_mask is None else votes[np.asarray(fit_mask, dtype=bool)]
    M = M == 1
    K = votes.shape[1]
    J = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            inter = int((M[:, i] & M[:, j]).sum())
            uni = int((M[:, i] | M[:, j]).sum())
            J[i, j] = inter / uni if uni else 0.0
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(K):
        for j in range(i + 1, K):
            if J[i, j] >= jaccard_min:
                parent[find(i)] = find(j)
    roots = {}
    for k in range(K):
        roots.setdefault(find(k), []).append(k)
    groups = sorted(roots.values(), key=lambda g: g[0])
    return {"groups": groups, "jaccard": J}


def merge_group_votes(votes: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """One column per group: malicious if any member votes malicious, benign if none does and at least one
    votes benign, abstain otherwise."""
    votes = np.asarray(votes)
    cols = []
    for g in groups:
        V = votes[:, g]
        cols.append(np.where((V == 1).any(axis=1), 1, np.where((V == 0).any(axis=1), 0, -1)))
    return np.column_stack(cols).astype(np.int8)


def dawid_skene_multistart(votes: np.ndarray, fit_mask: np.ndarray | None = None, n_iter: int = 5000,
                           tol: float = 1e-9, **kw) -> dict:
    """Dawid-Skene from the default start and from the malicious votes of each source; the solution with the
    highest log-likelihood on the fitted rows is kept (docs/ORACLES_EXTENDED.md, change 2)."""
    votes = np.asarray(votes)
    starts = [("default", None)] + [(f"source {k}", (votes[:, k] == 1).astype(np.float64))
                                    for k in range(votes.shape[1])]
    best, runs = None, []
    for name, q0 in starts:
        r = dawid_skene(votes, n_iter=n_iter, tol=tol, fit_mask=fit_mask, q0=q0, **kw)
        runs.append({"start": name, "log_likelihood": r["log_likelihood_final"], "iterations": r["iterations"],
                     "pi": r["pi"], "converged": r["converged"]})
        if best is None or r["log_likelihood_final"] > best[1]["log_likelihood_final"] + 1e-9:
            best = (name, r)
    out = dict(best[1])
    out["start"] = best[0]
    out["starts"] = runs
    return out


def aggregate_votes(votes: np.ndarray, method: str = "ds", fit_mask: np.ndarray | None = None,
                    names: list | None = None, group_jaccard: float = 0.5, **kw) -> dict:
    """Turn oracle votes into pseudo-labels and trust scores.

    ``majority``    equal-vote majority, trust = share of all K oracles voting for the label
    ``ds``          Dawid-Skene label model (fitted on the rows of ``fit_mask`` only), label = more
                    probable class, trust = its posterior
    ``ds_grouped``  the same label model after merging oracles whose malicious votes coincide
                    (Jaccard index >= ``group_jaccard`` on the fitted rows) into one source, with EM
                    started from several points (docs/ORACLES_EXTENDED.md, changes 1 and 2)

    Always returned: ``label`` (1, 0, or -1 when no label can be assigned), ``trust`` in
    [0, 1] and the vote counts ``n_mal``, ``n_ben``, ``n_abs``.
    """
    votes = np.asarray(votes)
    K = votes.shape[1]
    n_mal = (votes == 1).sum(axis=1).astype(np.int8)
    n_ben = (votes == 0).sum(axis=1).astype(np.int8)
    n_abs = (votes == -1).sum(axis=1).astype(np.int8)
    out = {"n_mal": n_mal, "n_ben": n_ben, "n_abs": n_abs, "K": K, "method": method}
    if method == "majority":
        out.update(majority_vote(votes))
        out["model"] = None
    elif method == "ds":
        ds = dawid_skene(votes, fit_mask=fit_mask, **kw)
        q = ds.pop("q")
        out["label"] = (q >= 0.5).astype(np.int8)
        out["trust"] = np.maximum(q, 1.0 - q).astype(np.float32)
        out["q"] = q
        out["model"] = ds
    elif method == "ds_grouped":
        names = list(names) if names is not None else [f"oracle {k}" for k in range(K)]
        grp = dependent_groups(votes, fit_mask, group_jaccard)
        Vg = merge_group_votes(votes, grp["groups"])
        ds = dawid_skene_multistart(Vg, fit_mask=fit_mask, **kw)
        q = ds.pop("q")
        out["label"] = (q >= 0.5).astype(np.int8)
        out["trust"] = np.maximum(q, 1.0 - q).astype(np.float32)
        out["q"] = q
        ds["sources"] = ["+".join(names[k] for k in g) for g in grp["groups"]]
        ds["group_jaccard"] = float(group_jaccard)
        ds["jaccard"] = {f"{names[i]}|{names[j]}": float(grp["jaccard"][i, j])
                         for i in range(K) for j in range(i + 1, K)}
        out["model"] = ds
    else:
        raise ValueError(f"unknown aggregation method '{method}'")
    return out


def high_trust_mask(agg: dict, threshold: float = 0.8) -> np.ndarray:
    return (agg["label"] != NO_MAJORITY) & (agg["trust"] >= threshold - 1e-9)


def refine(X: np.ndarray, agg: dict, fit_mask: np.ndarray, cfg: dict | None = None, seed: int = 0) -> dict:
    """Semi-supervised refinement of low-trust flows.

    A logistic-regression model is fitted, without sample weights, on (at most ``max_fit_rows``)
    high-trust flows inside ``fit_mask`` (training rows) and applied to every low-trust flow of
    all subsets (trust below ``high_trust``, or no majority under the equal-vote rule):

    * predicted P(malicious) > ``p_flip``            -> pseudo-label malicious
    * predicted P(malicious) < 1 - ``p_flip``        -> pseudo-label benign
    * otherwise the majority-vote label is kept; flows without a majority take the more
      probable class of the model.

    Every low-trust flow, relabelled or not, then carries the fixed weight ``refined_trust``;
    high-trust flows keep their own trust score as weight.  With ``enabled = False`` low-trust flows keep their
    majority label and their own trust (flows without a majority are dropped, weight 0).
    """
    cfg = dict(cfg or {})
    thr = float(cfg.get("high_trust", 0.8))
    p_flip = float(cfg.get("p_flip", 0.85))
    refined_trust = float(cfg.get("refined_trust", 0.6))
    enabled = bool(cfg.get("enabled", True))
    max_fit = int(cfg.get("max_fit_rows", 500_000))
    label0, trust0 = agg["label"], agg["trust"]
    high = high_trust_mask(agg, thr)
    low = ~high
    label = label0.copy()
    weight = trust0.astype(np.float32).copy()
    info = {"enabled": enabled, "high_trust_threshold": thr, "p_flip": p_flip, "refined_trust": refined_trust,
            "n_total": int(len(label)), "n_high": int(high.sum()), "n_low": int(low.sum()),
            "n_no_majority": int((label0 == NO_MAJORITY).sum())}
    if not enabled:
        weight[label0 == NO_MAJORITY] = 0.0
        label[label0 == NO_MAJORITY] = 0
        info.update({"n_relabelled": 0, "relabel_share": 0.0})
        return {"label": label.astype(np.int8), "weight": weight, "refined": np.zeros(len(label), bool),
                "p_refine": np.full(len(label), np.nan, np.float32), "info": info, "model": None}
    fit_idx = np.flatnonzero(high & fit_mask)
    rng = np.random.default_rng(seed)
    if len(fit_idx) > max_fit:
        fit_idx = rng.choice(fit_idx, size=max_fit, replace=False)
    Xf = np.nan_to_num(X[fit_idx], nan=0.0, posinf=0.0, neginf=0.0)
    yf = label0[fit_idx].astype(int)
    p = np.full(len(label), np.nan, dtype=np.float32)
    model = None
    if len(np.unique(yf)) < 2:
        log.warning("refinement skipped: high-trust training flows contain a single class")
        label[label0 == NO_MAJORITY] = 0
        weight[low] = np.where(label0[low] == NO_MAJORITY, 0.0, refined_trust)
        info.update({"n_relabelled": 0, "relabel_share": 0.0, "skipped": True})
    else:
        Xf = np.clip(Xf, -30, 30)
        mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
        sd[sd == 0] = 1.0                       # inputs are standardised on the fitting rows (helps the solver)
        model = LogisticRegression(C=float(cfg.get("C", 1.0)), max_iter=int(cfg.get("max_iter", 1000)))
        model.fit((Xf - mu) / sd, yf)
        info["lr_iterations"] = int(np.max(model.n_iter_))
        info["lr_converged"] = bool(np.max(model.n_iter_) < model.max_iter)
        li = np.flatnonzero(low)
        if len(li):
            Xl = np.clip(np.nan_to_num(X[li], nan=0.0, posinf=0.0, neginf=0.0), -30, 30)
            p[li] = model.predict_proba((Xl - mu) / sd)[:, 1]
            new = label0[li].copy()
            new[p[li] > p_flip] = 1
            new[p[li] < 1.0 - p_flip] = 0
            nomaj = label0[li] == NO_MAJORITY
            undecided = nomaj & (p[li] <= p_flip) & (p[li] >= 1.0 - p_flip)
            new[undecided] = (p[li][undecided] >= 0.5).astype(np.int8)
            changed = (new != label0[li]) & ~nomaj
            label[li] = new
            weight[li] = refined_trust
            info.update({"n_relabelled": int(changed.sum()),
                         "relabel_share": float(changed.sum() / max(1, (~nomaj).sum())),
                         "n_no_majority_assigned": int(nomaj.sum()), "fit_rows": int(len(fit_idx))})
        else:
            info.update({"n_relabelled": 0, "relabel_share": 0.0, "fit_rows": int(len(fit_idx))})
    return {"label": label.astype(np.int8), "weight": weight.astype(np.float32), "refined": low.copy(),
            "p_refine": p, "info": info, "model": model}

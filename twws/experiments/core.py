"""Core experiments: search, main result, oracle analysis, ablations, selection baselines,
leave-one-attack-out and partition protocols."""
from __future__ import annotations

import dataclasses

import itertools
import time
from dataclasses import asdict

import numpy as np

from ..metrics import binary_metrics, confusion, from_counts, per_family_recall, seed_summary
from ..tuning import search_twxgb
from ..weak import NO_MAJORITY, aggregate_votes
from .common import (Ctx, closest_to_mean, header, log, run_detector, run_seeds, strip_private, summarise,
                     with_bootstrap)


def _label_quality(y_true: np.ndarray, label: np.ndarray, family: np.ndarray | None = None) -> dict:
    """Quality of pseudo-labels against ground truth (rows without a label are counted separately)."""
    has = label != NO_MAJORITY
    c = confusion(y_true[has], label[has] == 1)
    n_mal = int((y_true == 1).sum())
    out = {**from_counts(**c), **c, "labelled_share": float(has.mean()), "n": int(len(label)),
           "pseudo_malicious_share": float((label[has] == 1).mean()) if has.any() else float("nan"),
           "true_malicious_share": float(y_true.mean()),
           # precision, recall, F1 and accuracy above refer to the flows that have a label; the next two count
           # every flow (a malicious flow without a label is a miss; a flow without a label is not correct)
           "recall_all_malicious": c["tp"] / n_mal if n_mal else float("nan"),
           "accuracy_all_flows": (c["tp"] + c["tn"]) / len(label) if len(label) else float("nan")}
    if family is not None:
        out["families"] = per_family_recall(y_true[has], label[has] == 1, family[has])
    return out


# --------------------------------------------------------------------------- search
def exp_search(ctx: Ctx) -> dict:
    prep = ctx.default_prep()
    res = {**header(ctx, "search"), **search_twxgb(prep, ctx.cfg, nthread=ctx.n_jobs)}
    ctx.ws.save_result("search", res)
    ctx.reset_best()
    return res


# --------------------------------------------------------------------------- main result
def exp_main(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    prep = ctx.default_prep()
    params = ctx.best_params()
    runs, summ = run_seeds(prep, params, cfg.experiments.seeds, cfg, ctx.n_jobs)
    rep = closest_to_mean(runs)
    te, tr = prep.test, prep.train
    res = {
        **header(ctx, "main"), "params": params, "params_source": ctx.params_source, "seeds": list(cfg.experiments.seeds),
        "runs": [strip_private(r) for r in runs], "summary": summ,
        "representative_seed": rep["seed"],
        "representative": {k: rep[k] for k in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy", "fpr",
                                                "auc_roc", "pr_auc", "threshold", "n_trees", "families"]},
        "bootstrap_ci_representative": with_bootstrap(rep, cfg),
        "pseudo_labels": {
            "aggregation": prep.opts.aggregation,
            "train_before_refinement": _label_quality(prep.y_true[tr], prep.agg["label"][tr], prep.family[tr]),
            "train_after_refinement": _label_quality(prep.y_true[tr], prep.y_pl[tr], prep.family[tr]),
            "test_after_refinement": _label_quality(prep.y_true[te], prep.y_pl[te], prep.family[te]),
        },
        "pipeline": prep.info,
        "feature_names": prep.names, "selected_features": prep.sel_names,
    }
    # the semi-supervised refinement of the first version, measured on the training flows when it is switched off
    # (docs/ORACLES_EXTENDED.md, change 3): labels before and after it, by transport protocol
    if not prep.opts.refine:
        from ..weak import refine as _refine
        rr = _refine(prep.Xs, prep.agg, fit_mask=tr,
                     cfg={"enabled": True, "p_flip": prep.opts.p_flip, "refined_trust": prep.opts.refined_trust,
                          "high_trust": prep.opts.high_trust,
                          "max_fit_rows": cfg.get_path("refinement.max_fit_rows", 500_000)}, seed=prep.opts.seed)
        tcp = ctx.ws.meta["proto"].astype(str).to_numpy() == "tcp"
        lab0, lab1 = prep.agg["label"], rr["label"]
        low_tr = rr["refined"] & tr
        res["refinement_check_train"] = {
            "before": _label_quality(prep.y_true[tr], lab0[tr], prep.family[tr]),
            "after": _label_quality(prep.y_true[tr], lab1[tr], prep.family[tr]),
            "after_tcp": _label_quality(prep.y_true[tr & tcp], lab1[tr & tcp]),
            "after_non_tcp": _label_quality(prep.y_true[tr & ~tcp], lab1[tr & ~tcp]),
            "low_trust": int(low_tr.sum()), "low_trust_non_tcp": int((low_tr & ~tcp).sum()),
            "to_malicious": int(((lab0 != 1) & (lab1 == 1) & tr).sum()),
            "to_malicious_benign": int(((lab0 != 1) & (lab1 == 1) & tr & (prep.y_true == 0)).sum()),
            "to_malicious_non_tcp": int(((lab0 != 1) & (lab1 == 1) & tr & ~tcp).sum()),
            "benign_non_tcp_flagged_after": float(((lab1 == 1) & tr & ~tcp & (prep.y_true == 0)).sum()
                                                  / max(1, (tr & ~tcp & (prep.y_true == 0)).sum())),
            "info": rr["info"]}
    # robustness: re-fit every learned component with a different seed
    rob = []
    for s in cfg.experiments.get("pipeline_seeds", []):
        p2 = ctx.prep.prepare(ctx.base_options(seed=int(s)))
        r = run_detector(p2, params, int(s), cfg, ctx.n_jobs)
        r["n_selected"] = int(p2.Xs.shape[1])
        rob.append(strip_private(r))
        log.info("pipeline seed %s: F1 %.4f with %d features", s, r["f1"], r["n_selected"])
    if rob:
        res["pipeline_seed_runs"] = rob
        res["pipeline_seed_summary"] = summarise(rob)
    ctx.ws.save_result("main", res)
    return res


# --------------------------------------------------------------------------- oracle analysis
def _oracle_stats(v: np.ndarray, y: np.ndarray) -> dict:
    voted = v >= 0
    mal, ben = v == 1, v == 0
    tp, fp = int((mal & (y == 1)).sum()), int((mal & (y == 0)).sum())
    fn, tn = int((ben & (y == 1)).sum()), int((ben & (y == 0)).sum())
    pos_voted, neg_voted = tp + fn, fp + tn
    return {"coverage": float(voted.mean()), "abstain_share": float((~voted).mean()),
            "malicious_vote_share": float(mal.mean()), "benign_vote_share": float(ben.mean()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "sensitivity_voted": tp / pos_voted if pos_voted else float("nan"),
            "false_alarm_voted": fp / neg_voted if neg_voted else float("nan"),
            "recall_all_malicious": tp / max(1, int((y == 1).sum())),
            "benign_flagged_share": fp / max(1, int((y == 0).sum())),
            "accuracy_voted": (tp + tn) / max(1, int(voted.sum())), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _pairwise(votes: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        both = (votes[:, i] >= 0) & (votes[:, j] >= 0)
        if both.sum() < 2:
            continue
        a, b = votes[both, i], votes[both, j]
        yy = y[both]
        agree = float((a == b).mean())
        pa, pb = a.mean(), b.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
        ea, eb = (a != yy).astype(float), (b != yy).astype(float)
        if ea.std() > 0 and eb.std() > 0:
            phi = float(np.corrcoef(ea, eb)[0, 1])
        else:
            phi = float("nan")
        both_wrong = float((ea * eb).mean())
        rows.append({"a": names[i], "b": names[j], "co_voted_share": float(both.mean()), "agreement": agree,
                     "kappa": float(kappa), "error_correlation": phi, "both_wrong_share": both_wrong,
                     "both_wrong_if_independent": float(ea.mean() * eb.mean())})
    return rows


def _joint_errors(votes: np.ndarray, y: np.ndarray) -> dict:
    """How often several oracles are wrong on the same flow, against the independence benchmark."""
    wrong = (votes >= 0) & (votes != y[:, None])
    n_wrong = wrong.sum(axis=1)
    K = votes.shape[1]
    p = wrong.mean(axis=0)
    # Poisson-binomial distribution of the number of wrong oracles under independence
    dist = np.zeros(K + 1)
    dist[0] = 1.0
    for pk in p:
        dist[1:] = dist[1:] * (1 - pk) + dist[:-1] * pk
        dist[0] *= (1 - pk)
    obs = np.bincount(n_wrong, minlength=K + 1) / len(y)
    return {"marginal_error_rate": p.tolist(), "observed": obs.tolist(), "independent": dist.tolist(),
            "share_ge2_wrong": float(obs[2:].sum()), "share_ge2_wrong_independent": float(dist[2:].sum()),
            "share_ge3_wrong": float(obs[3:].sum()), "share_ge3_wrong_independent": float(dist[3:].sum())}


def _trust_table(agg: dict, y: np.ndarray, method: str, n_oracles: int | None = None) -> list[dict]:
    """Trust score against the accuracy of the pseudo-label.

    Equal-vote majority: the trust is the share k/K of all K oracles that vote for the label, so the rows are the
    numbers k of agreeing oracles (code versions before 25 used the five-oracle shares 0.2, ..., 1.0 and missed the
    shares k/7 of seven oracles).  Label model: intervals of the posterior.  Every flow falls into exactly one row;
    a flow that would fall into none is counted in a row 'not binned', so that a binning error cannot hide flows."""
    lab, tr = agg["label"], agg["trust"]
    rows = []
    if method == "majority":
        K = int(n_oracles) if n_oracles else 5
        levels = [(-1.0, 0.0, "no majority")] + [((k - 0.5) / K, (k + 0.5) / K, f"{k}/{K}") for k in range(1, K + 1)]
    else:
        levels = [(0.5 - 1e-6, 0.6, "[0.5, 0.6)"), (0.6, 0.8, "[0.6, 0.8)"), (0.8, 0.95, "[0.8, 0.95)"),
                  (0.95, 0.99, "[0.95, 0.99)"), (0.99, 1.0 + 1e-6, "[0.99, 1]")]
    covered = np.zeros(len(lab), dtype=bool)
    for lo, hi, name in levels:
        m = (lab == NO_MAJORITY) if name == "no majority" else ((lab != NO_MAJORITY) & (tr >= lo) & (tr < hi))
        m &= ~covered
        covered |= m
        if not m.any():
            rows.append({"trust": name, "share": 0.0, "n": 0, "accuracy": float("nan"), "malicious_true_share": float("nan")})
            continue
        acc = float((lab[m] == y[m]).mean()) if name != "no majority" else float("nan")
        rows.append({"trust": name, "share": float(m.mean()), "n": int(m.sum()), "accuracy": acc,
                     "malicious_true_share": float(y[m].mean())})
    if not covered.all():
        m = ~covered
        rows.append({"trust": "not binned", "share": float(m.mean()), "n": int(m.sum()),
                     "accuracy": float((lab[m] == y[m]).mean()), "malicious_true_share": float(y[m].mean())})
    return rows


def exp_oracles(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    prep = ctx.default_prep()
    params = ctx.best_params()
    names = prep.oracle_names
    y, fam, V = prep.y_true, prep.family, prep.votes
    tr, te = prep.train, prep.test
    res = {**header(ctx, "oracles"), "oracles": names, "describe": prep.info["oracle_describe"]}
    res["per_oracle"] = {
        n: {"train": _oracle_stats(V[tr, j], y[tr]), "test": _oracle_stats(V[te, j], y[te]),
            "malicious_vote_share_by_family": {f: float((V[(fam == f), j] == 1).mean()) for f in sorted(set(fam))},
            "abstain_share_by_family": {f: float((V[(fam == f), j] == -1).mean()) for f in sorted(set(fam))}}
        for j, n in enumerate(names)}
    # the two oracle definitions of the first version, measured in the same way (not used for the labels)
    o0 = ctx.base_options()
    first = {}
    if "signature" in names and ctx.cfg.snort.get("attack_classtypes") is not None:
        first["signature (every alert, first version)"] = dataclasses.replace(o0, signature_any=True), "signature"
    if "domain" in names and ctx.cfg.oracles.domain.get("fanout_quantile") is not None:
        first[f"domain (fixed fan-out threshold {float(ctx.cfg.oracles.domain.fanout_min):g}, first version)"] = (
            dataclasses.replace(o0, domain_fanout_quantile=None, domain_fanout_min=float(ctx.cfg.oracles.domain.fanout_min)),
            "domain")
    if first:
        A0 = ctx.prep.stage_a(o0)
        res["first_version_oracles"] = {}
        for label, (oo, name) in first.items():
            v = ctx.prep._oracle_votes(name, oo, A0)["votes"]
            res["first_version_oracles"][label] = {
                "train": _oracle_stats(v[tr], y[tr]),
                "malicious_vote_share_by_family": {f: float((v[(fam == f)] == 1).mean()) for f in sorted(set(fam))}}
    res["pairwise_train"] = _pairwise(V[tr], y[tr], names)
    res["joint_errors_train"] = _joint_errors(V[tr], y[tr])
    res["joint_errors_train_malicious"] = _joint_errors(V[tr & (y == 1)], y[tr & (y == 1)])
    agg = {}
    ht = float(cfg.refinement.high_trust)
    # the label model as configured ('ds', or 'ds_grouped' of docs/ORACLES_EXTENDED.md) is stored under 'ds';
    # with 'ds_grouped' the ungrouped model is stored as well ('ds_ungrouped')
    lm = prep.opts.aggregation if prep.opts.aggregation in ("ds", "ds_grouped") else "ds"
    methods = [("majority", "majority"), ("ds", lm)] + ([("ds_ungrouped", "ds")] if lm == "ds_grouped" else [])
    for key_m, method in methods:
        a = aggregate_votes(V, method, fit_mask=tr, names=names, group_jaccard=prep.opts.group_jaccard)
        agg[key_m] = {
            "train": _label_quality(y[tr], a["label"][tr], fam[tr]), "test": _label_quality(y[te], a["label"][te], fam[te]),
            "trust_table_train": _trust_table({k: (v[tr] if isinstance(v, np.ndarray) else v) for k, v in a.items()},
                                              y[tr], method, n_oracles=V.shape[1]),
            "high_trust_share_train": float(((a["label"][tr] != NO_MAJORITY) & (a["trust"][tr] >= ht - 1e-9)).mean()),
            "high_trust_malicious_train": int(((a["label"][tr] == 1) & (a["trust"][tr] >= ht - 1e-9)).sum()),
            "method": method,
        }
        if method in ("ds", "ds_grouped"):
            m = a["model"]
            if method == "ds_grouped":
                from ..weak import dependent_groups, merge_group_votes
                grp = dependent_groups(V, tr, prep.opts.group_jaccard)["groups"]
                Vs, srcs = merge_group_votes(V, grp), m["sources"]
            else:
                Vs, srcs = V, list(names)
            true_rates = {n: _oracle_stats(Vs[:, j], y) for j, n in enumerate(srcs)}
            agg[key_m]["model"] = {
                "pi_estimated": m["pi"], "pi_true": float(y.mean()), "iterations": m["iterations"],
                "oracles": {n: {"sensitivity_estimated": m["sensitivity"][j], "sensitivity_true": true_rates[n]["sensitivity_voted"],
                                "false_alarm_estimated": m["false_alarm"][j], "false_alarm_true": true_rates[n]["false_alarm_voted"]}
                            for j, n in enumerate(srcs)}}
            for extra in ("start", "starts", "sources", "jaccard", "group_jaccard", "converged", "log_likelihood_final"):
                if extra in m:
                    agg[key_m]["model"][extra] = m[extra]
    res["aggregation"] = agg
    res["label_model_method"] = lm
    res["final_pseudo_labels_test"] = _label_quality(y[te], prep.y_pl[te], fam[te])
    # detector-level ablations: aggregation rule, leave-one-oracle-out, single oracle
    seeds = cfg.experiments.ablation_seeds
    variants = [("all oracles, label model", {"oracles": tuple(names), "aggregation": lm}),
                ("all oracles, equal-vote majority", {"oracles": tuple(names), "aggregation": "majority"})]
    if lm == "ds_grouped":
        variants.append(("all oracles, label model without grouping", {"oracles": tuple(names), "aggregation": "ds"}))
    for n in names:
        variants.append((f"without {n}", {"oracles": tuple(x for x in names if x != n), "aggregation": lm}))
    for n in names:
        variants.append((f"only {n}", {"oracles": (n,), "aggregation": "majority"}))
    abl = []
    for label, ch in variants:
        t0 = time.perf_counter()
        try:
            p2 = ctx.prep.prepare(ctx.base_options(**ch))
            ytr = p2.y_pl[p2.usable_train()]
            if len(np.unique(ytr)) < 2:
                abl.append({"variant": label, "status": "pseudo-labels contain a single class; detector not trainable",
                            "pseudo_labels_train": _label_quality(y[tr], p2.y_pl[tr])})
                continue
            runs, summ = run_seeds(p2, params, seeds, cfg, ctx.n_jobs)
            abl.append({"variant": label, "oracles": list(ch["oracles"]), "aggregation": ch["aggregation"],
                        "summary": summ, "n_features": int(p2.Xs.shape[1]),
                        "pseudo_labels_train": _label_quality(y[tr], p2.y_pl[tr]),
                        "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:  # keep going; the failure is reported
            log.exception("oracle ablation '%s' failed", label)
            abl.append({"variant": label, "status": f"failed: {exc}"})
    res["detector_ablation"] = abl
    res["ablation_seeds"] = list(seeds)
    ctx.ws.save_result("oracles", res)
    return res


# --------------------------------------------------------------------------- ablations
def ablation_variants(ctx: Ctx) -> list[dict]:
    """Every ablation as (group, name, preparation changes, detector-parameter changes)."""
    bp = ctx.best_params()
    k = int(ctx.cfg.selection.k)
    V = []

    def add(group, name, prep=None, params=None, note=""):
        V.append({"group": group, "name": name, "prep": prep or {}, "params": params or {}, "note": note})

    add("reference", "full method (selected configuration)")
    # ---- objective, built up step by step
    plain = {"focal_gamma": 0.0, "delta": None, "class_weight": "none", "use_trust": False, "warmup_rounds": 0}
    add("objective", "logistic loss, no weights", params=plain)
    add("objective", "+ trust weights", params={**plain, "use_trust": True})
    add("objective", "+ class weights (balanced)", params={**plain, "use_trust": True, "class_weight": "balanced"})
    add("objective", "+ class weights (reciprocal)", params={**plain, "use_trust": True, "class_weight": "reciprocal"})
    add("objective", "+ focal term (gamma = 2)", params={**plain, "use_trust": True, "class_weight": bp["class_weight"],
                                                           "focal_gamma": 2.0})
    for d in (0.1, 0.3, 1.0):
        add("objective", f"+ gradient clipping (delta = {d})",
            params={**plain, "use_trust": True, "class_weight": bp["class_weight"], "focal_gamma": 2.0, "delta": d,
                    "warmup_rounds": int(bp.get("warmup_rounds", 20))})
    add("objective", "selected configuration without trust weights", params={"use_trust": False})
    if bp.get("delta") is not None and int(bp.get("warmup_rounds", 0)) > 0:
        # the warm-up cap is defined relative to delta, so it only exists when clipping is selected
        add("objective", "selected configuration without warm-up capping", params={"warmup_rounds": 0})
    for g in (0.0, 1.0, 2.0):
        if float(bp.get("focal_gamma", -1)) != g:          # the selected value is the reference row
            add("objective", f"selected configuration, focal gamma = {g:g}", params={"focal_gamma": g})
    for cw in ("reciprocal", "balanced", "none"):
        if bp.get("class_weight") != cw:
            add("objective", f"selected configuration, class weighting {cw}", params={"class_weight": cw})
    for d in (0.1, 0.3, 1.0, None):
        if bp.get("delta") != d:
            add("objective", "selected configuration, " + (f"gradient clip delta = {d:g}" if d is not None else "no gradient clipping"),
                params={"delta": d})
    add("objective", "selected configuration, no row subsampling", params={"subsample": 1.0})
    # ---- feature selection
    add("selection", "no selection (all features)", prep={"selector": "none"})
    add("selection", f"mutual information only (top {k})", prep={"selector": "mi"})
    add("selection", f"random-forest ranking only (top {k})", prep={"selector": "rf"})
    add("selection", "stability threshold 0.6", prep={"theta": 0.6})
    add("selection", "stability threshold 1.0", prep={"theta": 1.0})
    add("selection", f"k = {k // 2}", prep={"k": k // 2})
    add("selection", f"k = {k + k // 2}", prep={"k": k + k // 2})
    add("selection", "without borderline re-evaluation", prep={"use_borderline": False})
    # ---- refinement
    if bool(ctx.cfg.get_path("refinement.enabled", True)):
        add("refinement", "without semi-supervised refinement", prep={"refine": False})
        add("refinement", "refinement threshold 0.75", prep={"p_flip": 0.75})
        add("refinement", "refinement threshold 0.95", prep={"p_flip": 0.95})
        add("refinement", "high-trust threshold 0.9", prep={"high_trust": 0.9})
        add("refinement", "high-trust threshold 0.6", prep={"high_trust": 0.6})
    else:   # extended oracle set (docs/ORACLES_EXTENDED.md, change 3): the refinement of the first version as variants
        add("refinement", "with the semi-supervised refinement of the first version", prep={"refine": True})
        add("refinement", "with refinement, threshold 0.95", prep={"refine": True, "p_flip": 0.95})
        add("refinement", "with refinement, high-trust threshold 0.6", prep={"refine": True, "high_trust": 0.6})
    # ---- preprocessing and feature subspaces
    add("preprocessing", "without imputation and scaling", prep={"impute": False, "scale": False})
    add("subspaces", "without latent features", prep={"use_latent": False})
    add("subspaces", "without host-level temporal features", prep={"temporal": "none"})
    add("subspaces", "statistical per-flow features only", prep={"temporal": "none", "use_latent": False})
    # ---- oracle settings (sensitivity)
    c = ctx.cfg.oracles
    add("oracle settings", "SYN-ratio threshold 0.35", prep={"stat_syn_ratio": 0.35})
    add("oracle settings", "SYN-ratio threshold 0.55", prep={"stat_syn_ratio": 0.55})
    add("oracle settings", "DBSCAN eps 1.2", prep={"cluster_eps": 1.2})
    add("oracle settings", "DBSCAN eps 2.0", prep={"cluster_eps": 2.0})
    add("oracle settings", "DBSCAN min_samples 10", prep={"cluster_min_samples": 10})
    add("oracle settings", "DBSCAN min_samples 50", prep={"cluster_min_samples": 50})
    if "transfer" in ctx.prep.available_oracles():   # without the transfer oracle these rows would repeat the reference
        add("oracle settings", "transfer band 0.5 / 0.5 (never abstains)", prep={"transfer_hi": 0.5, "transfer_lo": 0.5})
        add("oracle settings", "transfer band 0.7 / 0.3", prep={"transfer_hi": 0.7, "transfer_lo": 0.3})
    q = c.domain.get("fanout_quantile")
    if q is not None:
        for qq in (0.95, 0.999):
            if abs(float(q) - qq) > 1e-12:
                add("oracle settings", f"fan-out threshold at the {qq:g} quantile", prep={"domain_fanout_quantile": qq})
        add("oracle settings", f"fan-out threshold fixed at {c.domain.fanout_min:g} (first version)",
            prep={"domain_fanout_quantile": None, "domain_fanout_min": float(c.domain.fanout_min)})
    if ctx.cfg.snort.get("attack_classtypes") is not None:
        add("oracle settings", "signature oracle counts every Snort alert (first version)", prep={"signature_any": True})
    # sensitivity variants of the two further oracles, listed in docs/ORACLES_EXTENDED.md before any vote was computed
    avail = ctx.prep.available_oracles()
    if "rate" in avail:
        for qq in (0.99, 0.9999):
            add("oracle settings", f"repeated-connection threshold at the {qq:g} quantile", prep={"rate_quantile": qq})
    if "beacon" in avail:
        for dd in (0.05, 0.2):
            add("oracle settings", f"regularity dispersion at most {dd:g}", prep={"beacon_disp_max": dd})
        for nn in (5, 20):
            add("oracle settings", f"regularity with at least {nn} connections", prep={"beacon_min_conn": nn})
    if str(ctx.cfg.get_path("oracles.aggregation", "ds")) == "ds_grouped":
        add("oracle settings", "label model without merging dependent oracles", prep={"aggregation": "ds"})
    return V


def exp_ablations(ctx: Ctx, groups: list[str] | None = None) -> dict:
    cfg = ctx.cfg
    params = ctx.best_params()
    seeds = cfg.experiments.ablation_seeds
    name = "ablations"
    res = ctx.ws.load_result(name) if ctx.ws.has_result(name) else {**header(ctx, name), "variants": []}
    done = {(v["group"], v["name"]) for v in res["variants"] if "summary" in v}   # failed variants are retried
    variants = ablation_variants(ctx)
    order = {(v["group"], v["name"]): i for i, v in enumerate(variants)}   # rows keep this order after a retry
    for v in variants:
        if groups and v["group"] not in groups:
            continue
        if (v["group"], v["name"]) in done:
            continue
        res["variants"] = [e for e in res["variants"] if (e["group"], e["name"]) != (v["group"], v["name"])]
        t0 = time.perf_counter()
        entry = {"group": v["group"], "name": v["name"], "prep_changes": v["prep"], "param_changes": v["params"]}
        try:
            prep = ctx.prep.prepare(ctx.base_options(**v["prep"]))
            runs, summ = run_seeds(prep, {**params, **v["params"]}, seeds, cfg, ctx.n_jobs)
            entry.update({"summary": summ, "n_features": int(prep.Xs.shape[1]),
                          "mean_trees": float(np.mean([r["n_trees"] for r in runs])),
                          "clipped_share": float(np.mean([r["objective"]["clipped_share_mean"] or 0 for r in runs])),
                          "pseudo_labels_train": _label_quality(prep.y_true[prep.train], prep.y_pl[prep.train]),
                          "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:
            log.exception("ablation '%s' failed", v["name"])
            entry["status"] = f"failed: {exc}"
        res["variants"].append(entry)
        res["variants"].sort(key=lambda e: order.get((e["group"], e["name"]), len(order)))   # stable: references last
        res["seeds"] = list(seeds)
        ctx.ws.save_result(name, res)          # checkpoint after every variant
    # supervised reference: the same detector trained on ground-truth labels (uses labels!)
    if not any(v["name"].startswith("supervised reference") for v in res["variants"]):
        prep = ctx.default_prep()
        tr = np.flatnonzero(prep.train)
        sup = {"focal_gamma": 0.0, "delta": None, "class_weight": "balanced", "use_trust": False, "warmup_rounds": 0}
        runs = []
        for s in seeds:
            runs.append(run_detector(prep, {**params, **sup}, int(s), cfg, ctx.n_jobs, train_rows=tr,
                                     y_train=prep.y_true[tr], w_train=np.ones(len(tr), dtype=np.float32)))
        res["variants"].append({"group": "reference", "name": "supervised reference (ground-truth training labels)",
                                "summary": summarise(runs), "n_features": int(prep.Xs.shape[1]),
                                "note": "training labels are ground truth; early stopping and threshold still use "
                                        "pseudo-labelled validation rows"})
        runs = []
        for s in seeds:
            runs.append(run_detector(prep, {**params, **sup}, int(s), cfg, ctx.n_jobs, train_rows=tr,
                                     y_train=prep.y_true[tr], w_train=np.ones(len(tr), dtype=np.float32),
                                     truth_for_validation=True))
        res["variants"].append({"group": "reference", "name": "fully supervised reference (ground truth for training, "
                                                              "early stopping and threshold)",
                                "summary": summarise(runs), "n_features": int(prep.Xs.shape[1]),
                                "note": "upper bound: ground-truth labels everywhere"})
        ctx.ws.save_result(name, res)
    return res


# --------------------------------------------------------------------------- selection baselines
def exp_fs_compare(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    params = ctx.best_params()
    seeds = ctx.seeds("secondary")
    rows = []
    for key, label in [("proposed", "Proposed (stability-aware)"), ("pca", "PCA"), ("rfe", "RFE"),
                       ("boruta", "Boruta"), ("ae", "Autoencoder-based selection"), ("pca_mrmr", "Hybrid mRMR + PCA")]:
        t0 = time.perf_counter()
        try:
            prep = ctx.prep.prepare(ctx.base_options(selector=key))
            runs, summ = run_seeds(prep, params, seeds, cfg, ctx.n_jobs)
            rows.append({"key": key, "method": label, "summary": summ, "n_features": int(prep.Xs.shape[1]),
                         "selection_seconds": prep.info["selection"]["seconds"], "selector_info": prep.info["selection"].get("mode"),
                         "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:
            log.exception("selector '%s' failed", key)
            rows.append({"key": key, "method": label, "status": f"failed: {exc}"})
    res = {**header(ctx, "fs_compare"), "seeds": list(seeds), "rows": rows, "params": params}
    ctx.ws.save_result("fs_compare", res)
    return res


# --------------------------------------------------------------------------- leave-one-attack-out
def exp_loao(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    params = ctx.best_params()
    seeds = ctx.seeds("secondary")
    fams_present = sorted(set(ctx.ws.meta["family"].astype(str)) - {"Benign"})
    rows = []
    for fam in [f for f in cfg.experiments.loao_families if f in fams_present]:
        t0 = time.perf_counter()
        prep = ctx.prep.prepare(ctx.base_options(protocol=f"loao:{fam}"))
        runs = []
        for s in seeds:
            r = run_detector(prep, params, int(s), cfg, ctx.n_jobs)
            held = next((f for f in r["families"] if f["family"] == fam), None)
            runs.append({"seed": int(s), "held_out_recall": held["recall"] if held else float("nan"),
                         "held_out_support": held["support"] if held else 0, "benign_fpr": r["fpr"],
                         "recall_other_families": float(np.mean([f["recall"] for f in r["families"] if f["family"] != fam]))
                         if len(r["families"]) > 1 else float("nan"), "threshold": r["threshold"]})
        rows.append({"family": fam, "support": runs[0]["held_out_support"],
                     "held_out_recall": seed_summary([x["held_out_recall"] for x in runs]),
                     "benign_fpr": seed_summary([x["benign_fpr"] for x in runs]),
                     "recall_other_families": seed_summary([x["recall_other_families"] for x in runs]),
                     "runs": runs, "train_rows": int(prep.usable_train().sum()), "n_features": int(prep.Xs.shape[1]),
                     "pseudo_labels_test": _label_quality(prep.y_true[prep.test], prep.y_pl[prep.test],
                                                          prep.family[prep.test]),
                     "oracle_describe": prep.info.get("oracle_describe"),
                     "seconds": round(time.perf_counter() - t0, 1)})
        log.info("LOAO %s: recall %.3f", fam, rows[-1]["held_out_recall"]["mean"])
    res = {**header(ctx, "loao"), "seeds": list(seeds), "rows": rows, "params": params}
    ctx.ws.save_result("loao", res)
    return res


# --------------------------------------------------------------------------- partition protocols
def exp_protocols(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    params = ctx.best_params()
    seeds = ctx.seeds("secondary")
    fw = cfg.experiments.get("forward")
    forward = tuple((k, tuple(fw[k])) for k in ("train", "val", "test")) if fw else None
    days_present = set(ctx.ws.meta["day"].astype(str))
    plans = [("random", "Random by flow (first submission)", {}),
             ("conn", "Random by connection", {}),
             ("chrono", "Chronological within class and day", {})]
    if forward and all(set(dict(forward)[k]) & days_present for k in ("train", "val", "test")):
        plans.append(("forward", "Forward in time (whole days)", {"forward": forward}))
    rows = []
    for proto, label, extra in plans:
        for temporal in ("global", "none"):
            t0 = time.perf_counter()
            try:
                prep = ctx.prep.prepare(ctx.base_options(protocol=proto, temporal=temporal, **extra))
                if len(np.unique(prep.y_pl[prep.usable_train()])) < 2 or prep.test.sum() == 0:
                    rows.append({"protocol": proto, "label": label, "temporal": temporal,
                                 "status": "training pseudo-labels contain a single class or empty test split"})
                    continue
                runs, summ = run_seeds(prep, params, seeds, cfg, ctx.n_jobs)
                rows.append({"protocol": proto, "label": label, "temporal": temporal, "summary": summ,
                             "split_table": prep.info["split_table"], "n_features": int(prep.Xs.shape[1]),
                             "pseudo_labels_test": _label_quality(prep.y_true[prep.test], prep.y_pl[prep.test],
                                                                  prep.family[prep.test]),
                             "pseudo_labels_train": _label_quality(prep.y_true[prep.train], prep.y_pl[prep.train],
                                                                   prep.family[prep.train]),
                             "oracle_describe": prep.info.get("oracle_describe"),
                             "seconds": round(time.perf_counter() - t0, 1)})
            except Exception as exc:
                log.exception("protocol %s/%s failed", proto, temporal)
                rows.append({"protocol": proto, "label": label, "temporal": temporal, "status": f"failed: {exc}"})
    res = {**header(ctx, "protocols"), "seeds": list(seeds), "rows": rows, "params": params,
           "forward_days": {k: list(v) for k, v in forward} if forward else None}
    ctx.ws.save_result("protocols", res)
    return res

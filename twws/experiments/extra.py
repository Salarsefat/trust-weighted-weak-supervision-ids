"""Further experiments: end-to-end baselines, adversarial evasion, interpretability, online
operation and cross-dataset evaluation."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..baselines import BASELINES, baseline_threshold
from ..corpus import Workspace
from ..features import STAT_FEATURES, availability_time, build_base_features, feature_groups, statistical_features
from ..labels import load_schedule
from ..metrics import best_f1_threshold, binary_metrics, paired_bootstrap_f1, per_family_recall, seed_summary
from ..tuning import run_search, search_subsets
from ..utils import current_rss_mb, get_logger, peak_rss_mb
from .common import Ctx, closest_to_mean, header, log, run_detector, strip_private, summarise


# =========================================================================== baselines
def exp_baselines(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    seeds = list(cfg.experiments.seeds)
    name = "baselines"
    res = ctx.ws.load_result(name) if ctx.ws.has_result(name) else {**header(ctx, name), "models": {}}
    if all("summary" in res["models"].get(k, {}) for k in cfg.experiments.baselines):
        return res                                   # every model is complete (run-all calls this again)
    prep = ctx.default_prep()
    te = np.flatnonzero(prep.test)
    y_te = prep.y_true[te]
    # reference predictions of the proposed detector (first seed) for paired comparisons
    ref = run_detector(prep, ctx.best_params(), seeds[0], cfg, ctx.n_jobs, keep_scores=True)
    ref_pred = ref["_score"] > ref["threshold"]
    n_trials = int(cfg.experiments.baseline_trials)
    n_random = max(1, min(int(cfg.search.n_random), n_trials))
    tr_s, va_s = search_subsets(prep, cfg, int(cfg.search.seed))
    for key in cfg.experiments.baselines:
        if key in res["models"] and "summary" in res["models"][key]:
            continue
        if key not in BASELINES:
            res["models"][key] = {"status": "unknown baseline key"}
            continue
        cls = BASELINES[key]
        t0 = time.perf_counter()
        entry = {"name": cls.name, "space": cls.space}
        try:
            def objective(params, cls=cls):
                b = cls(params, seed=int(cfg.search.seed), cfg=cfg, n_jobs=ctx.n_jobs).fit(prep, tr_s, va_s, budget=0.5)
                return best_f1_threshold(prep.y_pl[va_s], b.score(prep, va_s))[1]

            sr = run_search(objective, cls.space, n_trials, n_random, int(cfg.search.seed))
            entry["search"] = {k: sr[k] for k in ("best_params", "best_value", "n_trials", "n_random", "seconds")}
            entry["search"]["trials"] = sr["trials"]
            runs, describe = [], None
            for s in seeds:
                b = cls(sr["best_params"], seed=int(s), cfg=cfg, n_jobs=ctx.n_jobs)
                b.fit(prep, np.flatnonzero(prep.usable_train()), np.flatnonzero(prep.val_a))
                thr, f1c = baseline_threshold(b, prep)
                score = b.score(prep, te)
                m = binary_metrics(y_te, score, thr)
                r = {"seed": int(s), **m, "val_b_pseudo_f1": float(f1c),
                     "families": per_family_recall(y_te, score > thr, prep.family[te]), "fit_seconds": round(b.fit_seconds, 1)}
                if s == seeds[0]:
                    r["paired_vs_proposed"] = paired_bootstrap_f1(y_te, ref_pred, score > thr,
                                                                  B=int(cfg.experiments.bootstrap),
                                                                  seed=int(cfg.get_path("experiments.bootstrap_seed", 0)))
                    describe = b.describe()
                runs.append(r)
                log.info("%s seed %s: P %.4f R %.4f F1 %.4f", cls.name, s, m["precision"], m["recall"], m["f1"])
            entry.update({"runs": runs, "summary": summarise(runs), "describe": describe,
                          "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:
            log.exception("baseline %s failed", key)
            entry["status"] = f"failed: {exc}"
        res["models"][key] = entry
        res["seeds"] = seeds
        res["protocol"] = {
            "input": "all features before selection (same matrix for every model)",
            "labels": "pseudo-labels of the default preparation; trust x class-balance sample weights",
            "trials": n_trials, "random_trials": n_random, "search_rows": int(len(tr_s)), "search_val_rows": int(len(va_s)),
            "search_budget": "half of the final training budget (rounds/epochs/trees)",
            "dl_max_train_rows": int(cfg.experiments.dl_max_train_rows), "dl_epochs": int(cfg.experiments.dl_epochs),
            "seq_window": int(cfg.experiments.seq_window),
        }
        ctx.ws.save_result(name, res)
    return res


# =========================================================================== adversarial evasion
def _controlled_endpoints(sched: dict) -> dict:
    """Per capture day, the addresses whose sending behaviour the adversary controls."""
    out = {}
    for day, d in sched["days"].items():
        s = set(d.get("attackers", []))
        if d.get("family") == "Bot":
            s |= set(d.get("victims_internal", []))
        out[day] = s
    return out


def _transform_stream(stream: pd.DataFrame, ctrl: np.ndarray, kind: str, level: float) -> pd.DataFrame:
    """Apply one evasion transformation to every attacker-controlled flow of the stream.

    padding   every packet carries ``level`` extra payload bytes (bytes/packet capped at 1500)
    dummy     ``level`` extra 40-byte packets are appended to each flow
    stretch   the connection lasts ``level`` times longer (same packets)
    throttle  the attacker spaces its connections ``level`` times further apart

    The counters of the reverse direction of the same connection are updated consistently.
    """
    s = stream.copy(deep=False)
    pk = s["pkts"].to_numpy(dtype=np.float64).copy()
    by = s["bytes"].to_numpy(dtype=np.float64).copy()
    if kind == "padding":
        room = np.maximum(1500.0 * pk - by, 0.0)
        add = np.where(ctrl, np.minimum(pk * level, room), 0.0)
        by = by + add
        s["bytes"] = by.astype(np.int64)
        s["pay"] = (s["pay"].to_numpy(dtype=np.float64) + add).astype(np.int64)
        s["cap"] = (s["cap"].to_numpy(dtype=np.float64) + add).astype(np.int64)
    elif kind == "dummy":
        add = np.where(ctrl, level, 0.0)
        s["pkts"] = (pk + add).astype(np.int64)
        s["bytes"] = (by + 40.0 * add).astype(np.int64)
        s["ack"] = (s["ack"].to_numpy(dtype=np.float64) + add).astype(np.int32)
    elif kind == "stretch":
        conn = s["conn"].to_numpy()
        hit = np.isin(conn, np.unique(conn[ctrl]))
        f = np.where(hit, level, 1.0)
        s["duration"] = s["duration"].to_numpy(dtype=np.float64) * f
        s["conn_duration"] = s["conn_duration"].to_numpy(dtype=np.float64) * f
        s["t_last"] = s["t_first"].to_numpy(dtype=np.float64) + s["duration"].to_numpy(dtype=np.float64)
    elif kind == "throttle":
        conn = s["conn"].to_numpy()
        ts = s["ts"].to_numpy(dtype=np.float64).copy()
        src = s["src_ip"].astype(str).to_numpy()
        day = s["day"].astype(str).to_numpy()
        idx = np.flatnonzero(ctrl)
        if len(idx):
            g = pd.DataFrame({"k": src[idx] + "|" + day[idx], "ts": ts[idx], "conn": conn[idx]})
            t0 = g.groupby("k")["ts"].transform("min").to_numpy()
            new_ts = t0 + level * (g["ts"].to_numpy() - t0)
            shift = pd.Series(new_ts - g["ts"].to_numpy(), index=g["conn"].to_numpy())
            shift = shift[~shift.index.duplicated()]
            d = pd.Series(conn).map(shift).fillna(0.0).to_numpy()
            s["ts"] = ts + d
            s["t_first"] = s["t_first"].to_numpy(dtype=np.float64) + d
            s["t_last"] = s["t_last"].to_numpy(dtype=np.float64) + d
    else:
        raise ValueError(kind)
    if kind in ("padding", "dummy"):
        # reverse-direction records carry the modified counters as rev_pkts / rev_bytes
        key = s["conn"].to_numpy() * 2 + s["dir"].to_numpy()
        lut_p = pd.Series(s["pkts"].to_numpy(), index=key)
        lut_b = pd.Series(s["bytes"].to_numpy(), index=key)
        lut_p = lut_p[~lut_p.index.duplicated()]
        lut_b = lut_b[~lut_b.index.duplicated()]
        other = s["conn"].to_numpy() * 2 + (1 - s["dir"].to_numpy())
        rp = pd.Series(other).map(lut_p)
        rb = pd.Series(other).map(lut_b)
        s["rev_pkts"] = rp.fillna(pd.Series(s["rev_pkts"].to_numpy())).to_numpy().astype(np.int64)
        s["rev_bytes"] = rb.fillna(pd.Series(s["rev_bytes"].to_numpy())).to_numpy().astype(np.int64)
    s["t_avail"] = availability_time(s)
    return s


def exp_adversarial(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    ac = cfg.experiments.adversarial
    prep = ctx.default_prep()
    params = ctx.best_params()
    seed = int(cfg.experiments.seeds[0])
    base = run_detector(prep, params, seed, cfg, ctx.n_jobs, keep_scores=True)
    model, thr = base["_model"], base["threshold"]
    rng = np.random.default_rng(seed)
    stream = ctx.ws.load_stream()
    meta = ctx.ws.meta
    sched = load_schedule(cfg.corpus.schedule)
    ctrl_sets = _controlled_endpoints(sched)
    lab = stream["label"].to_numpy()
    ctrl = np.zeros(len(stream), dtype=bool)
    for d, ips in ctrl_sets.items():
        # comparisons on the columns themselves (categories): no per-row string object for the whole stream
        m = (stream["day"].astype(str) == d).to_numpy() & (lab == 1)
        if m.any() and ips:
            ctrl |= m & stream["src_ip"].astype(str).isin(list(ips)).to_numpy()
    # evaluated rows: malicious test flows sent by an attacker-controlled endpoint
    rows = np.flatnonzero(prep.test & (prep.y_true == 1))
    srow = meta["stream_row"].to_numpy()
    rows = rows[ctrl[srow[rows]]]
    if len(rows) > int(ac.max_rows):
        rows = np.sort(rng.choice(rows, size=int(ac.max_rows), replace=False))
    need = np.zeros(len(stream), dtype=bool)
    need[srow[rows]] = True
    order_in_need = np.argsort(np.argsort(srow[rows]))        # build_base_features returns rows in stream order
    fam = prep.family[rows]
    cols = prep.feature_pipeline.base_columns

    def score_stream(s: pd.DataFrame) -> np.ndarray:
        Xb, _ = build_base_features(s, need)
        X = prep.feature_pipeline.transform(Xb[cols])
        return model.predict_proba(prep.selector.transform(X))[order_in_need]

    def rec(scores: np.ndarray) -> dict:
        det = scores > thr
        out = {"recall": float(det.mean()), "mean_score": float(scores.mean())}
        out["families"] = {f: {"support": int((fam == f).sum()), "recall": float(det[fam == f].mean())}
                           for f in sorted(set(fam))}
        return out

    clean = score_stream(stream)
    res = {**header(ctx, "adversarial"), "seed": seed, "threshold": thr, "evaluated_flows": int(len(rows)),
           "controlled_flows_in_stream": int(ctrl.sum()), "clean": rec(clean), "attacks": [],
           "check_clean_matches_cached": float(np.abs(clean - model.predict_proba(prep.Xs[rows])).max())}
    grid = [("padding", float(v)) for v in ac.pad_bytes] + [("dummy", float(v)) for v in ac.dummy_packets] \
        + [("stretch", float(v)) for v in ac.stretch] + [("throttle", float(v)) for v in ac.throttle]
    all_scores = [clean]
    for kind, level in grid:
        t0 = time.perf_counter()
        sc = score_stream(_transform_stream(stream, ctrl, kind, level))
        all_scores.append(sc)
        res["attacks"].append({"kind": kind, "level": level, **rec(sc), "seconds": round(time.perf_counter() - t0, 1)})
        log.info("evasion %s x%s: recall %.4f (clean %.4f)", kind, level, res["attacks"][-1]["recall"], res["clean"]["recall"])
    # per-flow best of grid: an adversary who can query the detector and pick, per flow, the
    # transformation that lowers its score most (interactions between flows are ignored)
    res["best_of_grid"] = rec(np.min(np.vstack(all_scores), axis=0))
    # ---- defence: augmentation of malicious pseudo-labelled training flows with random transformations
    tr = np.flatnonzero(prep.usable_train())
    mal = tr[prep.y_pl[tr] == 1]
    n_aug = min(len(mal), int(ac.get("augment_rows", 100_000)))
    aug = np.sort(rng.choice(mal, size=n_aug, replace=False)) if n_aug else mal
    raw = meta.iloc[aug].copy()
    pk = raw["pkts"].to_numpy(dtype=np.float64)
    pad = rng.choice(np.array(ac.pad_bytes, dtype=np.float64), size=len(raw))
    dum = rng.choice(np.array([0.0] + [float(v) for v in ac.dummy_packets]), size=len(raw))
    st = rng.choice(np.array([1.0] + [float(v) for v in ac.stretch]), size=len(raw))
    add = np.minimum(pk * pad, np.maximum(1500.0 * pk - raw["bytes"].to_numpy(dtype=np.float64), 0.0))
    raw["bytes"] = (raw["bytes"].to_numpy(dtype=np.float64) + add + 40.0 * dum).astype(np.int64)
    raw["pay"] = (raw["pay"].to_numpy(dtype=np.float64) + add).astype(np.int64)
    raw["cap"] = (raw["cap"].to_numpy(dtype=np.float64) + add).astype(np.int64)
    raw["pkts"] = (pk + dum).astype(np.int64)
    raw["ack"] = (raw["ack"].to_numpy(dtype=np.float64) + dum).astype(np.int32)
    raw["duration"] = raw["duration"].to_numpy(dtype=np.float64) * st
    Xb_aug = ctx.ws.X_base.iloc[aug].reset_index(drop=True).copy()
    S_new = statistical_features(raw.reset_index(drop=True))
    for c in STAT_FEATURES:
        if c in Xb_aug.columns:
            Xb_aug[c] = S_new[c].to_numpy()
    Xs_aug = prep.selector.transform(prep.feature_pipeline.transform(Xb_aug[cols]))
    from ..model import calibrate_threshold, train_twxgb

    Xtr = np.vstack([prep.Xs[tr], Xs_aug])
    ytr = np.concatenate([prep.y_pl[tr], np.ones(len(aug), dtype=np.int8)])
    wtr = np.concatenate([prep.w_pl[tr], prep.w_pl[aug]])
    va, vb, te = np.flatnonzero(prep.val_a), np.flatnonzero(prep.val_b), np.flatnonzero(prep.test)
    m2 = train_twxgb(Xtr, ytr, wtr, prep.Xs[va], prep.y_pl[va], params, seed=seed, nthread=ctx.n_jobs,
                     es_max_rows=int(cfg.model.es_max_rows))
    thr2, _ = calibrate_threshold(m2, prep.Xs[vb], prep.y_pl[vb])
    clean_test = binary_metrics(prep.y_true[te], m2.predict_proba(prep.Xs[te]), thr2)
    model_backup = model

    def score_stream2(s):
        Xb, _ = build_base_features(s, need)
        return m2.predict_proba(prep.selector.transform(prep.feature_pipeline.transform(Xb[cols])))[order_in_need]

    def rec2(scores):
        det = scores > thr2
        return {"recall": float(det.mean()),
                "families": {f: {"support": int((fam == f).sum()), "recall": float(det[fam == f].mean())}
                             for f in sorted(set(fam))}}

    hard = {"clean": rec2(score_stream2(stream)), "attacks": []}
    hs = []
    for kind, level in grid:
        sc = score_stream2(_transform_stream(stream, ctrl, kind, level))
        hs.append(sc)
        hard["attacks"].append({"kind": kind, "level": level, **rec2(sc)})
    hard["best_of_grid"] = rec2(np.min(np.vstack([score_stream2(stream)] + hs), axis=0))
    hard["test_metrics_clean"] = {k: clean_test[k] for k in ("precision", "recall", "f1", "accuracy", "fpr")}
    hard["augmented_rows"] = int(len(aug))
    hard["threshold"] = float(thr2)
    res["hardened"] = hard
    res["baseline_test_metrics_clean"] = {k: base[k] for k in ("precision", "recall", "f1", "accuracy", "fpr")}
    del model_backup
    ctx.ws.save_result("adversarial", res)
    return res


# =========================================================================== interpretability
def exp_interpret(ctx: Ctx) -> dict:
    import xgboost as xgb

    cfg = ctx.cfg
    prep = ctx.default_prep()
    params = ctx.best_params()
    seed = int(cfg.experiments.seeds[0])
    base = run_detector(prep, params, seed, cfg, ctx.n_jobs, keep_scores=True)
    model, thr = base["_model"], base["threshold"]
    rng = np.random.default_rng(seed)
    te = np.flatnonzero(prep.test)
    n = int(cfg.experiments.shap_rows)
    # stratified sample: half benign, half malicious with equal weight per family
    ben = te[prep.y_true[te] == 0]
    mal = te[prep.y_true[te] == 1]
    pick = [rng.choice(ben, size=min(len(ben), n // 2), replace=False)]
    fams = sorted(set(prep.family[mal]))
    for f in fams:
        idx = mal[prep.family[mal] == f]
        pick.append(rng.choice(idx, size=min(len(idx), max(1, (n // 2) // max(1, len(fams)))), replace=False))
    rows = np.sort(np.concatenate(pick))
    Xs = prep.Xs[rows]
    d = xgb.DMatrix(Xs, nthread=ctx.n_jobs)
    contrib = model.booster.predict(d, pred_contribs=True, iteration_range=(0, model.best_iteration + 1))
    phi, bias = contrib[:, :-1], contrib[:, -1]
    names = prep.sel_names
    margin = model.margin(Xs)
    additivity = float(np.abs(phi.sum(axis=1) + bias - margin).max())
    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(-mean_abs)
    groups = feature_groups(names)
    tot = float(mean_abs.sum()) or 1.0
    res = {**header(ctx, "interpret"), "seed": seed, "rows": int(len(rows)), "method":
           "exact TreeSHAP contributions to the raw score (xgboost pred_contribs)", "additivity_max_abs_error": additivity,
           "global": [{"feature": names[i], "mean_abs_shap": float(mean_abs[i]), "share": float(mean_abs[i] / tot),
                       "subspace": next(g for g, cols in groups.items() if names[i] in cols)} for i in order],
           "subspace_share_shap": {g: float(sum(mean_abs[names.index(c)] for c in cols) / tot) for g, cols in groups.items()},
           "subspace_count": {g: len(cols) for g, cols in groups.items()}}
    gain = model.booster.get_score(importance_type="total_gain")
    gsum = sum(gain.values()) or 1.0
    res["subspace_share_gain"] = {g: float(sum(gain.get(f"f{names.index(c)}", 0.0) for c in cols) / gsum)
                                  for g, cols in groups.items()}
    fam_rows = prep.family[rows]
    det = model.predict_proba(Xs) > thr
    res["per_family"] = {}
    for f in sorted(set(fam_rows)):
        m = fam_rows == f
        mphi = phi[m].mean(axis=0)
        top = np.argsort(-np.abs(mphi))[:6]
        res["per_family"][f] = {"rows": int(m.sum()), "detected_share": float(det[m].mean()),
                                "top": [{"feature": names[i], "mean_shap": float(mphi[i]),
                                         "mean_value_scaled": float(np.nanmean(Xs[m, i]))} for i in top]}
    # make the latent features readable: which inputs does each used latent dimension follow?
    lat = [c for c in names if c.startswith("z_")]
    probes = []
    if lat:
        tr = np.flatnonzero(prep.train)
        sub = rng.choice(tr, size=min(len(tr), 100_000), replace=False)
        stat_cols = [c for c in prep.names if not c.startswith(("t_", "z_"))]
        Sx = np.nan_to_num(prep.X[sub][:, [prep.names.index(c) for c in stat_cols]], nan=0.0)
        for c in sorted(lat, key=lambda c: -mean_abs[names.index(c)]):
            z = prep.X[sub, prep.names.index(c)]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.array([np.corrcoef(z, Sx[:, j])[0, 1] if Sx[:, j].std() > 0 and z.std() > 0 else 0.0
                              for j in range(Sx.shape[1])])
            r = np.nan_to_num(r)
            top = np.argsort(-np.abs(r))[:3]
            A = np.column_stack([Sx, np.ones(len(Sx))])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            r2 = 1.0 - float(((A @ coef - z) ** 2).sum() / max(((z - z.mean()) ** 2).sum(), 1e-12))
            probes.append({"latent": c, "share": float(mean_abs[names.index(c)] / tot), "linear_r2": r2,
                           "top_correlates": [{"feature": stat_cols[j], "r": float(r[j])} for j in top]})
    res["latent_probes"] = probes
    res["selected_features"] = names
    ctx.ws.save_result("interpret", res)
    return res


# =========================================================================== online operation
def _replay_worker(args):
    """Replay a share of the record stream at a fixed arrival rate and time every verdict."""
    (bundle_path, records_path, rate, seconds, batch_size, part, n_parts, sliding) = args
    import torch

    torch.set_num_threads(1)
    from ..online import OnlineDetector

    b = joblib.load(bundle_path)
    b["model"].nthread = 1
    det = OnlineDetector(b["fp"], b["selector"], b["model"], b["threshold"], b["base_columns"], sliding_window=sliding)
    rec = pd.read_parquet(records_path)
    if n_parts > 1:
        h = pd.util.hash_array(rec["src_ip"].astype(str).to_numpy()) % n_parts
        idx_all = np.flatnonzero(h == part)
    else:
        idx_all = np.arange(len(rec))
    n_total = int(rate * seconds)
    # arrival time of global record i is i / rate; this worker handles the records of its hosts
    idx_all = idx_all[idx_all < n_total]
    arrivals = idx_all / float(rate)
    rec = rec.iloc[idx_all].reset_index(drop=True)
    # warm-up outside the measurement (JIT, caches)
    det.process(rec.iloc[: min(len(rec), batch_size)])
    det = OnlineDetector(b["fp"], b["selector"], b["model"], b["threshold"], b["base_columns"], sliding_window=sliding)
    lat = np.empty(len(rec))
    i = 0
    t_start = time.perf_counter()
    n = len(rec)
    while i < n:
        now = time.perf_counter() - t_start
        j = int(np.searchsorted(arrivals, now, side="right"))
        if j <= i:
            time.sleep(max(0.0, min(arrivals[i] - now, 0.002)))
            continue
        j = min(j, i + batch_size)
        det.process(rec.iloc[i:j])
        lat[i:j] = (time.perf_counter() - t_start) - arrivals[i:j]
        i = j
    wall = time.perf_counter() - t_start
    return {"latency": lat, "wall": wall, "n": n, "timers": det.timers, "batches": det.batches,
            "rss_mb": current_rss_mb(), "state_records": det.engine.state_records(), "hosts": len(det.engine.hosts)}


def exp_online(ctx: Ctx) -> dict:
    import multiprocessing as mp

    from ..online import OnlineDetector, OnlineFeatureEngine

    cfg = ctx.cfg
    oc = cfg.experiments.online
    prep = ctx.default_prep()
    params = ctx.best_params()
    seed = int(cfg.experiments.seeds[0])
    base = run_detector(prep, params, seed, cfg, ctx.n_jobs, keep_scores=True)
    model, thr = base["_model"], base["threshold"]
    meta = ctx.ws.meta
    res = {**header(ctx, "online"), "seed": seed, "threshold": thr, "n_trees": model.n_trees,
           "n_selected_features": int(prep.Xs.shape[1]), "batch_size": int(oc.batch_size)}
    # ---- (a) the streaming engine reproduces the offline features (one capture day of the stream)
    stream = ctx.ws.load_stream()
    d0 = str(stream["day"].astype(str).value_counts().idxmin())
    sub = stream[stream["day"].astype(str) == d0].reset_index(drop=True)
    if len(sub) > 300_000:
        sub = sub.sort_values("t_avail", kind="stable").head(300_000).reset_index(drop=True)
    Xoff, _ = build_base_features(sub)
    o = np.lexsort((np.arange(len(sub)), sub["t_avail"].to_numpy()))
    ss = sub.iloc[o].reset_index(drop=True)
    for c in ("src_ip", "dst_ip"):
        ss[c] = ss[c].astype(str)
    eng = OnlineFeatureEngine()
    t0 = time.perf_counter()
    H = eng.batch(ss)
    dt = time.perf_counter() - t0
    refm = Xoff[eng.names].to_numpy(dtype=np.float64)[o]
    diff = np.where(np.isnan(H) & np.isnan(refm), 0.0, np.abs(H - refm))
    res["engine_equivalence"] = {"day": d0, "records": int(len(ss)), "max_abs_difference": float(np.nanmax(diff)),
                                 "nan_mismatches": int(np.isnan(diff).sum()),
                                 "engine_microseconds_per_record": 1e6 * dt / max(1, len(ss))}
    del stream, sub, Xoff
    # ---- records to replay: corpus rows in order of availability
    order = np.lexsort((np.arange(len(meta)), meta["t_avail"].to_numpy()))
    rec = meta.iloc[order].reset_index(drop=True)
    for c in ("src_ip", "dst_ip", "proto"):
        rec[c] = rec[c].astype(str)
    max_rate, secs = max(oc.rates), float(oc.seconds_per_rate)
    need_n = int(min(len(rec), max_rate * secs))
    rec = rec.iloc[:need_n]
    tmp = Path(tempfile.mkdtemp(prefix="twws_online_"))
    rec_path, bundle_path = tmp / "records.parquet", tmp / "bundle.joblib"
    rec.to_parquet(rec_path, index=False)
    joblib.dump({"fp": prep.feature_pipeline, "selector": prep.selector, "model": model, "threshold": thr,
                 "base_columns": prep.feature_pipeline.base_columns}, bundle_path)
    res["bundle_megabytes"] = os.path.getsize(bundle_path) / 1e6
    res["model_megabytes"] = len(model.booster.save_raw("ubj")) / 1e6
    # ---- (b) service time per micro-batch, single worker, records processed back to back
    import torch

    torch.set_num_threads(1)
    model.nthread = 1
    det = OnlineDetector(prep.feature_pipeline, prep.selector, model, thr, prep.feature_pipeline.base_columns)
    B = int(oc.batch_size)
    n_prof = min(len(rec), 60_000)
    det.process(rec.iloc[:B])
    det = OnlineDetector(prep.feature_pipeline, prep.selector, model, thr, prep.feature_pipeline.base_columns)
    stage = {"features": [], "normalise": [], "model": [], "total": []}
    rss0 = current_rss_mb()
    t_all = time.perf_counter()
    for i in range(0, n_prof, B):
        before = dict(det.timers)
        t0 = time.perf_counter()
        det.process(rec.iloc[i: i + B])
        stage["total"].append(time.perf_counter() - t0)
        for k in ("features", "normalise", "model"):
            stage[k].append(det.timers[k] - before[k])
    wall = time.perf_counter() - t_all
    res["service_time"] = {
        "records": int(n_prof), "batch_size": B,
        "per_batch_ms": {k: {"p50": 1e3 * float(np.percentile(v, 50)), "p95": 1e3 * float(np.percentile(v, 95)),
                             "mean": 1e3 * float(np.mean(v))} for k, v in stage.items()},
        "per_flow_ms_mean": 1e3 * wall / n_prof, "single_worker_capacity_flows_per_s": n_prof / wall,
        "rss_mb_before": rss0, "rss_mb_after": current_rss_mb(), "process_peak_rss_mb": peak_rss_mb(),
        "engine_state_records": det.engine.state_records(), "engine_hosts": len(det.engine.hosts)}
    model.nthread = ctx.n_jobs
    torch.set_num_threads(max(1, ctx.n_jobs))
    # ---- (c) load test: fixed arrival rate, latency from record availability to verdict
    cpu = os.cpu_count() or 1
    load = []
    mpc = mp.get_context("spawn")
    for w in [int(x) for x in oc.workers if int(x) <= cpu]:
        for rate in [int(r) for r in oc.rates]:
            if rate * secs > len(rec) + 1:
                continue
            args = [(str(bundle_path), str(rec_path), rate, secs, B, p, w, None) for p in range(w)]
            t0 = time.perf_counter()
            if w == 1:
                outs = [_replay_worker(args[0])]
            else:
                with mpc.Pool(w) as pool:
                    outs = pool.map(_replay_worker, args)
            lat = np.concatenate([o_["latency"] for o_ in outs])
            n_done = int(sum(o_["n"] for o_ in outs))
            wall = max(o_["wall"] for o_ in outs)
            achieved = n_done / wall
            row = {"workers": w, "offered_rate": rate, "seconds": secs, "records": n_done,
                   "achieved_rate": achieved, "saturated": bool(achieved < 0.97 * rate or np.percentile(lat, 95) > 1.0),
                   "latency_ms": {"mean": 1e3 * float(lat.mean()), "p50": 1e3 * float(np.percentile(lat, 50)),
                                  "p95": 1e3 * float(np.percentile(lat, 95)), "p99": 1e3 * float(np.percentile(lat, 99)),
                                  "max": 1e3 * float(lat.max())},
                   "rss_mb_per_worker": float(np.mean([o_["rss_mb"] for o_ in outs])),
                   "wall_seconds": round(time.perf_counter() - t0, 1)}
            load.append(row)
            log.info("online: %d worker(s) at %d flows/s -> P95 %.1f ms (achieved %.0f flows/s)", w, rate,
                     row["latency_ms"]["p95"], achieved)
    res["load_test"] = load
    res["latency_definition"] = ("time from the moment a finalised flow record is available to the detector until its "
                                 "verdict is emitted; the lifetime of the flow itself and Zeek's connection timeout are "
                                 "not included")
    # ---- (d) detection quality with sliding-window instead of training-frozen normalisation
    te = np.flatnonzero(prep.test)
    te = te[np.lexsort((te, meta["t_avail"].to_numpy()[te]))]
    Xb = ctx.ws.X_base.iloc[te][prep.feature_pipeline.base_columns].reset_index(drop=True)
    y_te, fam_te = prep.y_true[te], prep.family[te]
    frozen = model.predict_proba(prep.Xs[te])
    slide_rows = [{"normalization": "training-frozen", **{k: v for k, v in binary_metrics(y_te, frozen, thr).items()
                                                           if k in ("precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc")},
                   "families": per_family_recall(y_te, frozen > thr, fam_te)}]
    for W in [int(oc.window), int(oc.window) * 10]:
        d2 = OnlineDetector(prep.feature_pipeline, prep.selector, model, thr, prep.feature_pipeline.base_columns,
                            sliding_window=W)
        sc = np.empty(len(te), dtype=np.float32)
        for i in range(0, len(te), B):
            X = d2._transform_sliding(Xb.iloc[i: i + B])
            sc[i: i + B] = model.predict_proba(np.ascontiguousarray(prep.selector.transform(X), dtype=np.float32))
        slide_rows.append({"normalization": f"sliding window W = {W}",
                           **{k: v for k, v in binary_metrics(y_te, sc, thr).items()
                              if k in ("precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc")},
                           "families": per_family_recall(y_te, sc > thr, fam_te)})
    res["normalization_comparison"] = slide_rows
    # ---- (e) trace-driven cost of a source hold-down after a verdict
    hold = float(cfg.get_path("experiments.online.hold_down_s", 5.0))
    mt = meta.iloc[te]
    df = pd.DataFrame({"src": mt["src_ip"].astype(str).to_numpy(), "t_start": mt["ts"].to_numpy(dtype=float),
                       "t_verdict": mt["t_avail"].to_numpy(dtype=float), "y": y_te, "pred": frozen > thr})
    alarms = df[df["pred"]].sort_values("t_verdict")[["src", "t_verdict", "y"]].rename(columns={"y": "alarm_true"})
    flows = df.sort_values("t_start")
    if len(alarms):
        j = pd.merge_asof(flows, alarms, left_on="t_start", right_on="t_verdict", by="src", direction="backward",
                          allow_exact_matches=False, suffixes=("", "_alarm"))
        blocked = (j["t_start"] - j["t_verdict_alarm"]) <= hold
        blocked = blocked.fillna(False).to_numpy()
        yy = j["y"].to_numpy()
        res["hold_down"] = {"hold_seconds": hold, "benign_flows_blocked_share": float(blocked[yy == 0].mean()),
                            "malicious_flows_blocked_share": float(blocked[yy == 1].mean()),
                            "benign_flows": int((yy == 0).sum()), "malicious_flows": int((yy == 1).sum()),
                            "note": "flow-record replay of the test split; a flow counts as blocked when it starts "
                                    "within the hold-down period after a positive verdict for the same source"}
    ctx.ws.save_result("online", res)
    return res


# =========================================================================== cross-dataset
def exp_cross_dataset(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    from ..stages import _abs

    src_root = _abs(cfg.transfer.work_dir)
    res = {**header(ctx, "cross_dataset")}
    if not (src_root / "corpus" / "meta.parquet").exists():
        res["status"] = "not run: no second corpus found under " + str(cfg.transfer.work_dir)
        ctx.ws.save_result("cross_dataset", res)
        return res
    ws2 = Workspace(src_root)
    params = ctx.best_params()
    seeds = ctx.seeds("secondary")
    names = [n for n in ctx.prep.available_oracles() if n != "transfer"]
    prep = ctx.prep.prepare(ctx.base_options(oracles=tuple(names)))
    cols = prep.feature_pipeline.base_columns
    y2 = (ws2.meta["label"].to_numpy() == 1).astype(int)
    fam2 = ws2.meta["family"].astype(str).to_numpy()
    Xb2 = ws2.X_base[cols]
    X2 = prep.selector.transform(prep.feature_pipeline.transform(Xb2))
    # variant: median/IQR re-estimated on the (unlabelled) target traffic
    pre = prep.feature_pipeline.pre
    med, iqr = pre.median_.copy(), pre.iqr_.copy()
    A = Xb2.to_numpy(dtype=np.float64)
    m2 = np.nanmedian(A, axis=0)
    q1, q3 = np.nanpercentile(A, [25, 75], axis=0)
    i2 = q3 - q1
    pre.median_ = np.where(np.isfinite(m2), m2, med)
    pre.iqr_ = np.where(np.isfinite(i2) & (i2 > 0), i2, iqr)
    try:
        X2r = prep.selector.transform(prep.feature_pipeline.transform(Xb2))
    finally:
        pre.median_, pre.iqr_ = med, iqr
    rows_f, rows_r, src_runs = [], [], []
    for s in seeds:
        r = run_detector(prep, params, int(s), cfg, ctx.n_jobs, keep_scores=True)
        model, thr = r["_model"], r["threshold"]
        src_runs.append(strip_private(r))
        for X, store in ((X2, rows_f), (X2r, rows_r)):
            sc = model.predict_proba(np.ascontiguousarray(X, dtype=np.float32))
            m = binary_metrics(y2, sc, thr)
            m["families"] = per_family_recall(y2, sc > thr, fam2)
            m["seed"] = int(s)
            store.append(m)
    res.update({
        "source": cfg.corpus.name, "target": load_schedule(cfg.transfer.schedule)["dataset"],
        "oracles_used": names, "note": "the transfer oracle is left out so that no label of the target dataset "
                                       "influences training",
        "target_rows": int(len(y2)), "target_malicious_share": float(y2.mean()),
        "target_families": pd.Series(fam2).value_counts().to_dict(),
        "source_test": summarise(src_runs), "target_frozen_statistics": summarise(rows_f),
        "target_recalibrated_statistics": summarise(rows_r), "seeds": seeds})
    ctx.ws.save_result("cross_dataset", res)
    return res


# =========================================================================== QoS under enforcement
def exp_qos(ctx: Ctx) -> dict:
    """Quality of service under enforcement, by trace-driven emulation (twws/qos.py).

    Every record of the stream of the retained captures (not only the corpus sample) is classified by the detector
    of the selected configuration; its verdicts, the decision latency measured in the load test of ``online`` and a
    source hold-down drive an edge filter whose delivered traffic is measured against ground truth.
    """
    from ..qos import emulate, latency_for_rate

    cfg = ctx.cfg
    qc = cfg.experiments.get("qos") or {}
    speeds = [float(x) for x in qc.get("speeds", [1, 2, 5, 10, 20])]
    holds = [float(x) for x in qc.get("holds", [1, 5, 30])]
    hold = float(cfg.get_path("experiments.online.hold_down_s", 5.0))
    chunk = int(qc.get("chunk_rows", 400_000))
    prep = ctx.default_prep()
    params = ctx.best_params()
    seed = int(cfg.experiments.seeds[0])
    base = run_detector(prep, params, seed, cfg, ctx.n_jobs, keep_scores=True)
    model, thr = base["_model"], base["threshold"]
    cols = prep.feature_pipeline.base_columns
    t_start = time.perf_counter()
    stream = ctx.ws.load_stream()
    day_cat = stream["day"].cat
    day_codes = day_cat.codes.to_numpy()
    src_codes = stream["src_ip"].cat.codes.to_numpy().astype(np.int64)
    days, per_day = [], []
    for code, name in enumerate(day_cat.categories):
        idx = np.flatnonzero(day_codes == code)
        if len(idx) == 0:
            continue
        df = stream.iloc[idx].reset_index(drop=True)
        X, _ = build_base_features(df)
        score = np.empty(len(df), dtype=np.float32)
        for a in range(0, len(df), chunk):
            Xs = prep.selector.transform(prep.feature_pipeline.transform(X[cols].iloc[a:a + chunk]))
            score[a:a + chunk] = model.predict_proba(np.ascontiguousarray(Xs, dtype=np.float32))
        lab = df["label"].to_numpy().astype(np.int8)
        d = {"day": str(name), "src": src_codes[idx], "ts": df["ts"].to_numpy(dtype=np.float64),
             "t_avail": df["t_avail"].to_numpy(dtype=np.float64), "duration": df["duration"].to_numpy(dtype=np.float64),
             "bytes": df["bytes"].to_numpy(dtype=np.float64), "pkts": df["pkts"].to_numpy(dtype=np.float64),
             "label": lab, "pred": score > thr}
        days.append(d)
        known = lab >= 0
        m = binary_metrics(lab[known].astype(int), score[known], thr) if known.any() and len(np.unique(lab[known])) > 1 \
            else {}
        span = float(max(d["t_avail"].max(), (d["ts"] + d["duration"]).max()) - d["ts"].min())
        per_day.append({"day": str(name), "records": int(len(df)), "malicious": int((lab == 1).sum()),
                        "unknown": int((lab < 0).sum()), "span_seconds": span,
                        "benign_bytes": float(d["bytes"][lab == 0].sum()),
                        "flagged": int(d["pred"].sum()),
                        **{k: m.get(k) for k in ("precision", "recall", "f1", "fpr") if k in m}})
        log.info("qos: %s: %d records classified (%d flagged)", name, len(df), int(d["pred"].sum()))
        del df, X
    del stream
    classify_seconds = time.perf_counter() - t_start
    lab_all = np.concatenate([d["label"] for d in days])
    pred_all = np.concatenate([d["pred"] for d in days])
    known = lab_all >= 0
    tp = int(((pred_all == 1) & (lab_all == 1)).sum())
    fp = int(((pred_all == 1) & (lab_all == 0)).sum())
    fn = int(((pred_all == 0) & (lab_all == 1)).sum())
    tn = int(((pred_all == 0) & (lab_all == 0)).sum())
    verdicts = {"records": int(len(lab_all)), "records_with_ground_truth": int(known.sum()), "tp": tp, "fp": fp,
                "fn": fn, "tn": tn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                "fpr": fp / max(1, fp + tn),
                "note": "all records of the stream, of which the corpus is a sample; about 5% of them were training "
                        "flows of the detector"}
    span_total = float(sum(x["span_seconds"] for x in per_day))
    rate1 = float(len(lab_all)) / max(span_total, 1e-9)        # records per second reaching the monitor
    online = ctx.ws.load_result("online") if ctx.ws.has_result("online") else {}
    load_rows = online.get("load_test", []) or []

    def lat_for(rate):
        if not load_rows:                     # no load test in this workspace: verdicts without decision latency
            return {"rate": rate, "within_capacity": True, "measured_rate": None, "p50_ms": 0.0, "p95_ms": 0.0}
        return latency_for_rate(load_rows, rate)

    rows = []
    for attack in (False, True):
        for sp in speeds:
            lat = lat_for(rate1 * sp)
            lat_s = lat.get("p95_ms", 0.0) / 1e3 if lat.get("within_capacity") else float("nan")
            if not np.isfinite(lat_s):
                rows.append({"speed": sp, "attack": attack, "hold_s": hold, "record_rate": rate1 * sp,
                             "status": "beyond the measured capacity of the decision path"})
                continue
            r = emulate(days, speed=sp, latency_s=lat_s, hold_s=hold, attack=attack)
            r.update({"record_rate": rate1 * sp, "latency": lat})
            rows.append(r)
            log.info("qos: %s, speed %g: preservation %.4f, availability %.4f, SLA %.4f", "attack" if attack else
                     "no attack", sp, r["throughput_preservation"], r["availability"], r["sla_violation_share"])
    hold_rows = []
    lat1 = lat_for(rate1)
    lat1_s = lat1.get("p95_ms", 0.0) / 1e3 if lat1.get("within_capacity") else 0.0
    for attack in (False, True):
        for h in holds:
            r = emulate(days, speed=1.0, latency_s=lat1_s, hold_s=h, attack=attack)
            r["latency"] = lat1
            hold_rows.append(r)
    res = {**header(ctx, "qos"), "seed": seed, "threshold": float(thr), "hold_s": hold, "speeds": speeds,
           "holds": holds, "record_rate": rate1, "span_seconds": span_total, "per_day": per_day,
           "verdicts": verdicts, "rows": rows, "hold_rows": hold_rows,
           "latency_source": "load test of the online experiment" if load_rows else "no load test available (latency 0)",
           "classify_seconds": round(classify_seconds, 1),
           "definition": {
               "enforcement": "a malicious verdict blocks the source address of the record for the hold-down period at "
                              "an edge filter; overlapping blocks are merged",
               "verdict_time": "availability of the record plus the P95 decision latency measured in the load test at "
                               "the smallest sustained arrival rate at or above that of the emulated load (with the "
                               "number of workers that gave the lowest P95 there; qos.latency_for_rate)",
               "compression": "all times of the trace divided by the speed factor; latency and hold-down not compressed",
               "throughput_preservation": "delivered / offered benign bytes",
               "availability": "share of benign records of which no part is dropped",
               "packet_loss_ratio": "dropped / offered benign packets",
               "sla_violation_share": "share of one-second intervals with offered benign traffic in which less than 99% "
                                      "of it is delivered",
               "attack_bytes_dropped_share": "dropped / offered attack bytes"}}
    ctx.ws.save_result("qos", res)
    return res


def exp_trust(ctx: Ctx) -> dict:
    """Trust calibration of the two aggregation rules on the training split (table 'trust').

    Recomputes only what that table needs: the oracle votes on the default partition (without the feature
    pipeline, which no oracle uses) and the two aggregations, with the binning of the majority trust by the number
    of agreeing oracles (``core._trust_table``, code version 25).  The oracle experiment of earlier code versions
    binned it at the five-oracle shares 0.2, ..., 1.0 and so missed the shares k/7 of seven oracles.  The quality of
    the pseudo-labels is written as well, so that the report can check that they are those of the oracle experiment.
    """
    from ..corpus import TEST, TRAIN, make_split
    from ..weak import aggregate_votes
    from .core import _label_quality, _trust_table

    o = ctx.base_options()
    meta = ctx.ws.meta
    forward = {k: list(v) for k, v in o.forward} if o.forward else None
    split = make_split(meta, o.protocol, o.split_seed, forward)
    A = {"split": split}
    available = ctx.prep.available_oracles()
    use = [n for n in o.oracles if n in available]
    t0 = time.perf_counter()
    V = np.column_stack([ctx.prep._oracle_votes(n, o, A)["votes"] for n in use]).astype(np.int8)
    y = (meta["label"].to_numpy() == 1).astype(np.int8)
    fam = meta["family"].astype(str).to_numpy()
    tr, te = split == TRAIN, split == TEST
    lm = o.aggregation if o.aggregation in ("ds", "ds_grouped") else "ds"
    res = {**header(ctx, "trust"), "oracles": use, "n_oracles": len(use), "aggregation": {}}
    for key_m, method in (("majority", "majority"), ("ds", lm)):
        a = aggregate_votes(V, method, fit_mask=tr, names=use, group_jaccard=o.group_jaccard)
        res["aggregation"][key_m] = {
            "method": method,
            "train": _label_quality(y[tr], a["label"][tr], fam[tr]),
            "test": _label_quality(y[te], a["label"][te], fam[te]),
            "trust_table_train": _trust_table({k: (v[tr] if isinstance(v, np.ndarray) else v) for k, v in a.items()},
                                              y[tr], method, n_oracles=V.shape[1]),
        }
    res["seconds"] = time.perf_counter() - t0
    log.info("trust tables in %.0f s (%d oracles)", res["seconds"], len(use))
    ctx.ws.save_result("trust", res)
    return res


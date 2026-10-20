"""Data stages: capture retrieval, extraction, corpus assembly and the transfer-oracle source model."""
from __future__ import annotations

from pathlib import Path
import inspect

import joblib
import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .corpus import Workspace, assemble_stream, connection_table
from .features import STAT_FEATURES, statistical_features
from .labels import estimate_utc_offset, label_flows, load_schedule
from .oracles import TransferOracle
from .utils import StageTimer, ensure_dir, environment_record, get_logger, write_json

log = get_logger("twws.stages")


def workspace(cfg) -> Workspace:
    """Workspace of a configuration.  ``capture_dir`` (optional) names the workspace whose captures and Zeek/Snort
    output are used; configurations that differ only after extraction share them this way."""
    root = Path(cfg.work_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    cap = cfg.get("capture_dir")
    if cap:
        cap = Path(cap)
        if not cap.is_absolute():
            cap = REPO_ROOT / cap
    # ``corpus_work_dir`` (optional): a configuration that changes only the oracles or the aggregation uses the
    # corpus of another workspace (read only) and writes its own results, cache and calibration
    cw = cfg.get("corpus_work_dir")
    if cw:
        cw = Path(cw)
        if not cw.is_absolute():
            cw = REPO_ROOT / cw
    return Workspace(root, cap, corpus_root=cw)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- real data
def stage_fetch(cfg, list_only: bool = False, local_root: str | None = None, days: list[str] | None = None) -> dict:
    """Retrieve (or index) the retained captures and write the file-level manifest.

    * no ``local_root``: ranged retrieval of the selected captures from the public bucket;
    * ``local_root`` holding the daily archives (``<root>/<day>/pcap.zip``): the same selection
      and extraction, read from disk;
    * ``local_root`` holding extracted captures (``<root>/<day>/<capture>`` or files named after
      the day): the captures are indexed in place.
    """
    from .fetch import compare_inventory, fetch_ids2018, find_local_archive, manifest_from_local

    ws = workspace(cfg)
    data_dir = ensure_dir(ws.capture_root / "data")
    if local_root and not any(find_local_archive(local_root, d) for d in cfg.corpus.days):
        man = manifest_from_local(cfg, local_root, data_dir)
        return {"mode": "local captures", "captures": int(len(man)), "bytes": int(man["file_size"].sum())}
    res = fetch_ids2018(cfg, data_dir, list_only=list_only, days=days, local_archives=local_root)
    sel = res.members[res.members["role"] != "unselected"] if len(res.members) else res.members
    out = {"mode": ("list" if list_only else "retrieve") + (" (local archives)" if local_root else " (public bucket)"),
           "members_listed": int(len(res.members)), "captures_selected": int(len(sel)),
           "bytes_selected": int(sel["file_size"].sum()) if len(sel) else 0}
    if len(sel):
        out["per_day"] = {d: {"captures": int((sel["day"] == d).sum()),
                              "gigabytes": round(float(sel.loc[sel["day"] == d, "file_size"].sum()) / 1e9, 2)}
                          for d in sorted(sel["day"].unique())}
    live = data_dir / "bucket_listing_live.csv"
    pinned = _abs(cfg.corpus.pinned_inventory) if cfg.corpus.get("pinned_inventory") else None
    if live.exists() and pinned is not None and pinned.exists():
        cmp = compare_inventory(pd.read_csv(live), pinned)
        cmp.to_csv(data_dir / "inventory_check.csv", index=False)
        both = cmp[cmp["_merge"] == "both"]
        out["inventory_objects_compared"] = int(len(both))
        out["inventory_all_match"] = bool(both["size_match"].all() and both["etag_match"].all())
    write_json(data_dir / "fetch_report.json", out)
    return out


def stage_ingest(cfg, local_root: str | None = None, delete_pcaps: bool = False, days: list[str] | None = None) -> dict:
    """Day by day: retrieve the retained captures, run Zeek and Snort, optionally delete the captures.

    Keeps the disk footprint at one day of captures; the manifest (with SHA-256 digests) is kept.
    """
    import shutil

    ws = workspace(cfg)
    out = {}
    for day in (days or list(cfg.corpus.days)):
        f = stage_fetch(cfg, local_root=local_root, days=[day])
        n_day = int(f.get("per_day", {}).get(day, {}).get("captures", f.get("captures", 0)) or 0)
        if n_day == 0:   # every capture day has victims: an empty selection means the listing was not understood
            raise RuntimeError(f"no capture of {day} was selected; check the archive listing (member names, "
                               "victim addresses of the schedule) before continuing")
        e = stage_extract(cfg)
        out[day] = {"fetch": f.get("per_day", {}).get(day, f), "flows": e.get("zeek_flows_new")}
        pdir = ws.capture_root / "data" / "pcap" / day
        if delete_pcaps and pdir.is_dir():
            shutil.rmtree(pdir)
            out[day]["captures_deleted"] = True
    write_json(ws.capture_root / "data" / "ingest_report.json", out)
    return out


def stage_extract(cfg) -> dict:
    """Zeek and Snort over every capture of the manifest.

    Both tools run in one pool of workers, largest capture first, so that the two runs over the largest
    capture proceed side by side instead of one after the other.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .extract import _process_capture, check_captures, record_zeek_rows, zeek_binary, zeek_jobs, zeek_version
    from .snort import _process as snort_process
    from .snort import record_snort_rows, snort_jobs, snort_setup
    from .utils import worker_count

    ws = workspace(cfg)
    data_dir = ws.capture_root / "data"
    extract_dir, snort_dir = ws.capture_root / "extract", ws.capture_root / "snort"
    man = pd.read_csv(data_dir / "manifest.csv")
    timer = StageTimer()
    with timer.stage("check captures"):
        checks = check_captures(man, data_dir, extract_dir, snort_dir)
    with_snort = "signature" in cfg.oracles.enabled
    with timer.stage("zeek and snort"):
        tasks = [("zeek", j) for j in zeek_jobs(cfg, man, data_dir, extract_dir)]
        sinfo = None
        if with_snort:
            sinfo = snort_setup(cfg, snort_dir)
            tasks += [("snort", j) for j in snort_jobs(cfg, man, data_dir, snort_dir, sinfo)]
        tasks.sort(key=lambda t: -(os.path.getsize(t[1][0]) if os.path.exists(t[1][0]) else 0))
        n_jobs = worker_count(cfg.get_path("extract.n_jobs"))
        zrows, srows = [], []
        if tasks:
            log.info("running Zeek on %d and Snort on %d captures with %d workers, largest capture first",
                     sum(k == "zeek" for k, _ in tasks), sum(k == "snort" for k, _ in tasks), n_jobs)
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(_process_capture if kind == "zeek" else snort_process, job): kind for kind, job in tasks}
                for f in as_completed(futs):
                    r = f.result()
                    if futs[f] == "zeek":
                        zrows.append(r)
                        log.info("zeek done: %s/%s %d flows in %.0f s", r["day"], r["capture"], r["flows"], r["seconds"])
                    else:
                        srows.append(r)
                        log.info("snort done: %s/%s %d alerts in %.0f s", r["day"], r["capture"], r["alerts"], r["seconds"])
        rep = record_zeek_rows(zrows, extract_dir, zeek_binary(cfg))
        record_snort_rows(srows, snort_dir)
    info = {"zeek_version": zeek_version(cfg), "captures": int(len(man))}
    if with_snort:
        sinfo["captures"] = srows
        info["snort"] = sinfo
    info["timing"] = timer.as_list()
    info["zeek_flows_new"] = int(rep["flows"].sum()) if len(rep) else 0
    if len(checks):
        info["captures_checked_now"] = int(len(checks))
        info["captures_repaired_now"] = int(checks["repaired_now"].sum())
    write_json(ws.capture_root / "extract" / "extract_info.json", info)
    return info


def stage_corpus(cfg) -> dict:
    """Assemble the stream (de-duplication, labels, Snort flag) and the evaluation corpus."""
    from .extract import deduplicate, load_flows
    from .snort import alerts_to_connections, load_alerts
    from .extract import connection_ids

    from .features import WINDOWS

    cfg_windows = {str(k): float(v) for k, v in dict(cfg.features.get("windows") or WINDOWS).items()}
    if cfg_windows != WINDOWS:   # the kernel, the oracles and the streaming engine are written for these windows
        raise ValueError(f"features.windows = {cfg_windows} differs from the implemented windows {WINDOWS}; "
                         "other windows are not supported")
    if cfg.get("corpus_work_dir"):
        raise RuntimeError("this configuration reads the corpus of another workspace (corpus_work_dir); build the "
                           "corpus with that workspace's configuration")
    ws = workspace(cfg)
    timer = StageTimer()
    with timer.stage("load flows"):
        flows = load_flows(ws.capture_root / "extract", list(cfg.corpus.days))
    if len(flows) == 0:
        raise RuntimeError(f"no extracted flows under {ws.capture_root / 'extract'}: run fetch/ingest and extract first")
    missing = sorted(set(cfg.corpus.days) - set(flows["day"].astype(str).unique()))
    if missing:
        raise RuntimeError(f"no extracted flows for the capture day(s) {missing}; complete the extraction first "
                           "(or remove the day from corpus.days and state this in the paper)")
    snort_conn = None
    with timer.stage("assemble stream"):
        tol = float(cfg.corpus.get("dedup_tolerance_s", 1.0))
        flows, dstats = deduplicate(flows, tol)          # the table as loaded is released here
        stream, info = assemble_stream(flows, cfg.corpus.schedule, None, tol, deduplicated=dstats)
    del flows
    if "signature" in cfg.oracles.enabled and (ws.capture_root / "snort" / "alerts").exists():
        with timer.stage("map snort alerts"):
            from .snort import attack_alert_mask, rule_classtypes
            from .utils import read_json

            alerts = load_alerts(ws.capture_root / "snort")
            conns = connection_table(stream)
            slack = float(cfg.snort.get("alert_slack_s", 1.0))

            def flag(al: pd.DataFrame) -> tuple[np.ndarray, int]:
                if not len(al):
                    return np.zeros(len(stream), dtype=bool), 0
                sc = alerts_to_connections(al, conns, slack)
                hit = set(sc.loc[sc["snort_alert"], "conn"].tolist())
                return stream["conn"].isin(hit).to_numpy(), len(hit)

            # every alert (definition of the first version) and alerts of the attack classes only (signature oracle)
            any_flag, n_any = flag(alerts)
            stream["snort_alert_any"] = any_flag
            classes = cfg.snort.get("attack_classtypes")
            if classes is None:
                stream["snort_alert"] = any_flag
                n_att, n_att_alerts = n_any, int(len(alerts))
            else:
                einfo_p = ws.capture_root / "extract" / "extract_info.json"
                snort_info = (read_json(einfo_p).get("snort") or {}) if einfo_p.exists() else {}
                rule_files = [x["file"] for x in snort_info.get("rule_files", [])]
                if not rule_files:
                    raise RuntimeError(f"the rule files of the Snort run are not recorded in {einfo_p}: the classes of "
                                       "the alerts cannot be determined (snort.attack_classtypes)")
                cmap = rule_classtypes(rule_files)
                mask = attack_alert_mask(alerts, cmap, classes)
                att_flag, n_att = flag(alerts[mask])
                stream["snort_alert"] = att_flag
                n_att_alerts = int(mask.sum())
                if len(alerts):
                    cls = alerts["sid"].map(cmap).where(alerts["gid"].to_numpy() == 1, "preprocessor or decoder event")
                    info["snort_alerts_by_class"] = {str(k): int(v) for k, v in cls.fillna("rule without class").value_counts().items()}
                info["snort_attack_classtypes"] = list(classes)
            info["snort_alerts_total"] = int(len(alerts))
            info["snort_alerts_attack_classes"] = n_att_alerts
            info["snort_alert_connections"] = int(n_att)
            info["snort_alert_connections_any"] = int(n_any)
            info["snort_alert_flows"] = int(stream["snort_alert"].sum())
            info["snort_alert_flows_any"] = int(stream["snort_alert_any"].sum())
            if len(alerts):
                top = alerts.groupby(["sid", "msg"]).size().sort_values(ascending=False).head(25)
                info["snort_top_signatures"] = [{"sid": int(s), "msg": m, "alerts": int(n)} for (s, m), n in top.items()]
    with timer.stage("save stream"):
        ws.save_stream(stream, info)
    with timer.stage("sample corpus + base features"):
        rep = ws.build_corpus(stream, cfg)
    rep["timing"] = timer.as_list()
    rep["environment"] = environment_record()
    write_json(ws.corpus_dir / "corpus_info.json", rep)
    return rep


def stage_transfer(cfg) -> dict:
    """Train the transfer-oracle random forest on the labelled source dataset (CICIDS2017)."""
    from .extract import deduplicate, load_flows

    ws = workspace(cfg)
    src_root = _abs(cfg.transfer.work_dir)
    flows = load_flows(src_root / "extract")
    if len(flows) == 0:
        raise FileNotFoundError(f"no extracted flows of the labelled source dataset under {src_root / 'extract'}; "
                                "run fetch/extract with configs/ids2017.yaml first (or leave the transfer oracle out)")
    flows, dstats = deduplicate(flows, 1.0)
    sched = load_schedule(cfg.transfer.schedule)
    off, diag = estimate_utc_offset(flows, sched)
    lab = label_flows(flows, sched, off)
    keep = lab["label"].to_numpy() >= 0
    S = statistical_features(flows.loc[keep]).to_numpy(dtype=np.float32)
    y = lab["label"].to_numpy()[keep].astype(int)
    cap = cfg.transfer.get("max_source_rows")
    if cap and len(y) > cap:
        idx = np.random.default_rng(0).choice(len(y), size=int(cap), replace=False)
        S, y = S[idx], y[idx]
    orc = TransferOracle(hi=cfg.oracles.transfer.hi, lo=cfg.oracles.transfer.lo, columns=list(STAT_FEATURES))
    orc.fit_source(S, y, n_estimators=int(cfg.oracles.transfer.n_estimators),
                   max_depth=int(cfg.oracles.transfer.max_depth), seed=0, n_jobs=int(cfg.compute.n_jobs))
    out_dir = ensure_dir(ws.root / "transfer")
    joblib.dump(orc, out_dir / "transfer_rf.joblib")
    info = {**orc.info, "utc_offset_hours": int(off), "deduplication": dstats,
            "families": lab.loc[keep, "family"].value_counts().to_dict()}
    write_json(out_dir / "transfer_info.json", info)
    return info


# --------------------------------------------------------------------------- oracle settings fixed on CICIDS2017
CALIBRATION_QUANTILES = (0.95, 0.99, 0.995, 0.999)


def stage_pairs(cfg) -> dict:
    """Pair statistics of the corpus rows for the oracles 'rate' and 'beacon' (docs/ORACLES_EXTENDED.md),
    computed over the stored stream; no label is read."""
    return workspace(cfg).build_pairs()


def stage_oracle_check(cfg) -> dict:
    """Behaviour of the two revised oracle rules on the labelled second corpus (CICIDS2017, ``transfer.work_dir``).

    Diagnostic only: the settings of both rules are fixed a priori and use no label (``snort.attack_classtypes``,
    ``oracles.domain.fanout_quantile``); nothing here is fed back into the configuration.

    * fan-out rule of the domain oracle: for several quantiles, the threshold is that quantile of the fan-out of the
      originator flows of the training subset (as in the pipeline), and precision, recall and F1-score of the rule's
      malicious votes are measured on that subset; the same for the fixed threshold of the first version;
    * signature oracle: the same measures when every Snort alert counts (first version) and when only alerts of the
      attack classes count.

    Written to ``<transfer.work_dir>/calibration/oracle_check.json``.
    """
    from .corpus import TRAIN, make_split

    src = Workspace(_abs(cfg.transfer.work_dir))
    meta, aux = src.meta, src.aux
    y = meta["label"].to_numpy()
    split = make_split(meta, "random", 42)
    tr = (split == TRAIN) & (y >= 0)
    yt = y[tr] == 1

    def scores(flag: np.ndarray) -> dict:
        f = flag[tr]
        tp, fp, fn = int((f & yt).sum()), int((f & ~yt).sum()), int((~f & yt).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        return {"flagged": int(f.sum()), "tp": tp, "fp": fp, "precision": prec, "recall": rec, "f1": f1,
                "benign_flagged_share": float(fp / max(1, int((~yt).sum())))}

    fan = aux["aux_fanout_5m"].to_numpy(dtype=np.float64)
    orig = meta["dir"].to_numpy() == 0
    ref = tr & orig & np.isfinite(fan)
    rows = []
    for q in sorted(set(CALIBRATION_QUANTILES) | {float(cfg.oracles.domain.get("fanout_quantile") or 0.99)}):
        thr = float(np.quantile(fan[ref], q))
        rows.append({"quantile": q, "threshold": thr, **scores(orig & (fan >= thr))})
    fixed = float(cfg.oracles.domain.fanout_min)
    out = {"corpus": str(src.root), "training_rows": int(tr.sum()), "malicious_share": float(yt.mean()),
           "fanout": {"quantiles": rows, "configured_quantile": cfg.oracles.domain.get("fanout_quantile"),
                      "fixed_threshold_first_version": {"threshold": fixed, **scores(orig & (fan >= fixed))}}}
    if "snort_alert" in meta.columns:
        out["signature"] = {"attack_classes": scores(meta["snort_alert"].to_numpy(dtype=bool))}
        if "snort_alert_any" in meta.columns:
            out["signature"]["every_alert_first_version"] = scores(meta["snort_alert_any"].to_numpy(dtype=bool))
    write_json(ensure_dir(src.root / "calibration") / "oracle_check.json", out)
    return out


# --------------------------------------------------------------------------- weak labels before the experiments
def stage_label_check(cfg) -> dict:
    """Quality of the weak labels of the default pipeline on the training subset, before the experiments run.

    The oracle votes are computed exactly as in the pipeline (``Preparer``) and combined with both aggregation
    rules; ground truth is read only to measure them, as in the oracle experiment, whose numbers the paper reports.
    It is a check that the labels are not degenerate (a broken transfer model, an oracle that never votes) before
    days of computation are spent; nothing here changes a setting.  Written to
    ``<work_dir>/calibration/label_check.json``.
    """
    from .corpus import TRAIN, make_split
    from .experiments.common import Ctx
    from .experiments.core import _label_quality, _oracle_stats
    from .weak import aggregate_votes

    ctx = Ctx(cfg)
    o = ctx.base_options()
    meta = ctx.ws.meta
    forward = {k: list(v) for k, v in o.forward} if o.forward else None
    split = make_split(meta, o.protocol, o.split_seed, forward)     # as in stage A of the pipeline
    names = ctx.prep.available_oracles()
    runs = {n: ctx.prep._oracle_votes(n, o, {"split": split}) for n in names}
    V = np.column_stack([runs[n]["votes"] for n in names])
    y = meta["label"].to_numpy()
    fam = meta["family"].astype(str).to_numpy()
    tr = split == TRAIN
    fams = sorted(set(fam[tr]))
    out = {"rows": int(len(meta)), "train_rows": int(tr.sum()), "true_malicious_share_train": float(y[tr].mean()),
           "oracles": names, "describe": {n: runs[n]["describe"] for n in names},
           "per_oracle_train": {n: _oracle_stats(V[tr, j], y[tr]) for j, n in enumerate(names)},
           "malicious_vote_share_by_family_train": {
               n: {f: float((V[tr & (fam == f), j] == 1).mean()) for f in fams} for j, n in enumerate(names)},
           "aggregation_train": {}}
    # per attack (the schedule's attack names within a family): which attacks the oracles and the pseudo-labels cover
    att = meta["attack"].astype(str).to_numpy() if "attack" in meta.columns else np.full(len(meta), "", dtype=object)
    mal_tr = tr & (y == 1)
    attacks = sorted(set(att[mal_tr]))
    out["attacks_train"] = {a: {"family": str(fam[mal_tr & (att == a)][0]), "flows": int((mal_tr & (att == a)).sum())}
                            for a in attacks}
    out["malicious_vote_share_by_attack_train"] = {
        n: {a: float((V[mal_tr & (att == a), j] == 1).mean()) for a in attacks} for j, n in enumerate(names)}
    out["pseudo_label_recall_by_attack_train"] = {}
    # 'ds' holds the label model as configured ('ds' or 'ds_grouped', docs/ORACLES_EXTENDED.md); with 'ds_grouped'
    # the ungrouped model is added as 'ds_ungrouped'.  The EM diagnostics below concern the ungrouped model.
    lm = o.aggregation if o.aggregation in ("ds", "ds_grouped") else "ds"
    out["label_model_method"] = lm
    methods = [("majority", "majority"), ("ds", lm)] + ([("ds_ungrouped", "ds")] if lm == "ds_grouped" else [])
    for key_m, method in methods:
        a = aggregate_votes(V, method, fit_mask=tr, names=names, group_jaccard=o.group_jaccard)
        rec = _label_quality(y[tr], a["label"][tr], fam[tr])
        if method in ("ds", "ds_grouped") and a.get("model"):
            rec["model"] = {k: v for k, v in a["model"].items()
                            if k in ("pi", "sensitivity", "false_alarm", "n_iter", "iterations", "sources", "start",
                                     "starts", "jaccard", "converged")}
        out["aggregation_train"][key_m] = rec
        out["pseudo_label_recall_by_attack_train"][key_m] = {
            x: float((a["label"][mal_tr & (att == x)] == 1).mean()) for x in attacks}
    if lm == "ds_grouped":
        # (i) the label model without merging, fitted with the same starting points and iteration limit as the
        # merged one: is the difference made by the merging or by the EM budget? (ii) how the configured model
        # relates to the repeated-connection oracle on the training flows (docs/ORACLES_EXTENDED.md)
        from .weak import dawid_skene_multistart
        ms = dawid_skene_multistart(V, tr)
        lab_ms = (ms["q"] >= 0.5).astype(np.int8)
        lab_u = aggregate_votes(V, "ds", fit_mask=tr)["label"]
        out["ds_ungrouped_multistart"] = {
            **{k: v for k, v in _label_quality(y[tr], lab_ms[tr], fam[tr]).items() if k != "families"},
            "start": ms["start"], "iterations": int(ms["iterations"]), "converged": bool(ms["converged"]),
            "log_likelihood": float(ms["log_likelihood_final"]),
            "starts": [{k: r[k] for k in ("start", "log_likelihood", "iterations", "converged")} for r in ms["starts"]],
            "labels_equal_ds_ungrouped_train": bool((lab_ms[tr] == lab_u[tr]).all()),
            "label_changes_vs_ds_ungrouped_train": int((lab_ms[tr] != lab_u[tr]).sum())}
        # the five oracles of the first version with the same merging rule (the gain is not the merging alone)
        first = [k for k, n in enumerate(names) if n in ("stat", "signature", "cluster", "transfer", "domain")]
        if 0 < len(first) < len(names):
            a5 = aggregate_votes(V[:, first], lm, fit_mask=tr, names=[names[k] for k in first],
                                 group_jaccard=o.group_jaccard)
            out["ds_grouped_first_version_oracles"] = {
                **{k: v for k, v in _label_quality(y[tr], a5["label"][tr], fam[tr]).items() if k != "families"},
                "sources": a5["model"]["sources"], "start": a5["model"]["start"]}
        if "rate" in names:
            from .oracles import ABS as _ABS
            lab_g = aggregate_votes(V, lm, fit_mask=tr, names=names, group_jaccard=o.group_jaccard)["label"]
            r_mal = V[:, names.index("rate")] == 1
            r_abs = V[:, names.index("rate")] == _ABS
            g_mal = lab_g == 1
            ym = y == 1
            out["label_model_vs_rate_train"] = {
                "malicious_flows": int((tr & ym).sum()),
                "malicious_flows_same_decision": int((tr & ym & (g_mal == r_mal)).sum()),
                "malicious_labelled_not_rate": int((tr & ym & g_mal & ~r_mal).sum()),
                "malicious_rate_not_labelled": int((tr & ym & ~g_mal & r_mal).sum()),
                "labelled_malicious": int((tr & g_mal).sum()), "rate_malicious": int((tr & r_mal).sum()),
                "labelled_not_rate": int((tr & g_mal & ~r_mal).sum()),
                "labelled_not_rate_benign": int((tr & g_mal & ~r_mal & ~ym).sum()),
                "labelled_where_rate_abstains": int((tr & g_mal & r_abs).sum()),
                "rate_alone": {k: v for k, v in _label_quality(y[tr], r_mal[tr].astype(np.int8), fam[tr]).items()
                               if k != "families"}}
    pat, cnt = np.unique(V[tr], axis=0, return_counts=True)
    out["vote_patterns_train"] = [
        {"votes": dict(zip(names, pat[i].tolist())), "rows": int(cnt[i]),
         "true_malicious_share": float(y[tr][(V[tr] == pat[i]).all(axis=1)].mean())}
        for i in np.argsort(-cnt)[:20]]
    # which vote patterns the label model labels malicious (every pattern with at least one such training flow)
    ds = aggregate_votes(V, "ds", fit_mask=tr)
    lab_tr = ds["label"][tr]
    rows_by_pat = []
    for i in np.argsort(-cnt):
        m = (V[tr] == pat[i]).all(axis=1)
        n_mal = int((lab_tr[m] == 1).sum())
        if n_mal:
            rows_by_pat.append({"votes": dict(zip(names, pat[i].tolist())), "rows": int(cnt[i]), "ds_malicious": n_mal,
                                "true_malicious_share": float(y[tr][m].mean())})
    out["ds_malicious_by_vote_pattern_train"] = rows_by_pat
    if "stat" in names and "transfer" in names:
        vs, vt = V[tr, names.index("stat")], V[tr, names.index("transfer")]
        agree = (vs == 1) & (vt == 1)
        one_abstain = ((vs == 1) & (vt == -1)) | ((vs == -1) & (vt == 1))
        mal = lab_tr == 1
        out["ds_stat_transfer_train"] = {"ds_malicious": int(mal.sum()), "agree_rows": int(agree.sum()),
                                         "ds_malicious_agree": int((mal & agree).sum()),
                                         "ds_malicious_one_abstains": int((mal & ~agree & one_abstain).sum()),
                                         "ds_malicious_other": int((mal & ~agree & ~one_abstain).sum())}
    # sensitivity of the pseudo-labels to the iteration limit of EM: the same fit with ten times as many iterations
    from .weak import dawid_skene
    n_default = int(inspect.signature(dawid_skene).parameters["n_iter"].default)
    ds10 = aggregate_votes(V, "ds", fit_mask=tr, n_iter=10 * n_default)
    out["ds_iteration_check"] = {"iterations_default": int(ds["model"]["iterations"]), "limit_default": n_default,
                                 "iterations_10x": int(ds10["model"]["iterations"]), "limit_10x": 10 * n_default,
                                 "label_changes_all_rows": int((ds10["label"] != ds["label"]).sum()),
                                 "pi_default": float(ds["model"]["pi"]), "pi_10x": float(ds10["model"]["pi"]),
                                 "max_abs_posterior_change": float(np.abs(ds10["q"] - ds["q"]).max())}
    l0, l1 = ds["label"][tr], ds10["label"][tr]
    q10 = _label_quality(y[tr], l1, fam[tr])
    out["ds_iteration_check"].update({
        "label_changes_train": int((l0 != l1).sum()),
        "to_malicious_train": int(((l0 != 1) & (l1 == 1)).sum()),
        "to_malicious_benign_train": int(((l0 != 1) & (l1 == 1) & (y[tr] == 0)).sum()),
        "to_benign_train": int(((l0 == 1) & (l1 != 1)).sum()),
        "precision_10x_train": q10["precision"], "recall_10x_train": q10["recall_all_malicious"],
        "recall_by_attack_10x_train": {x: float((ds10["label"][mal_tr & (att == x)] == 1).mean()) for x in attacks}})
    if "stat" in names:
        st_tr = V[tr, names.index("stat")] == 1
        out["ds_iteration_check"].update({
            "stat_flagged_train": int(st_tr.sum()),
            "malicious_10x_train": int((l1 == 1).sum()),
            "malicious_10x_stat_flagged_train": int(((l1 == 1) & st_tr).sum()),
            "malicious_10x_not_stat_flagged_train": int(((l1 == 1) & ~st_tr).sum()),
            "stat_flagged_not_malicious_10x_train": int(((l1 != 1) & st_tr).sum())})
    # the same check for every oracle set of the oracle ablation that uses the label model: does EM stop at the
    # limit, and which pseudo-labels does it reach when it is allowed to converge
    variants = {"all oracles": list(range(len(names)))}
    for j, n in enumerate(names):
        variants[f"without {n}"] = [k for k in range(len(names)) if k != j]
    rows = []
    for vname, cols in variants.items():
        a0 = ds if vname == "all oracles" else aggregate_votes(V[:, cols], "ds", fit_mask=tr)
        a1 = ds10 if vname == "all oracles" else aggregate_votes(V[:, cols], "ds", fit_mask=tr, n_iter=10 * n_default)
        lab0, lab1 = a0["label"][tr], a1["label"][tr]
        rows.append({"variant": vname, "iterations": int(a0["model"]["iterations"]),
                     "at_limit": bool(a0["model"]["iterations"] >= n_default),
                     "iterations_10x": int(a1["model"]["iterations"]),
                     "converged_10x": bool(a1["model"]["iterations"] < 10 * n_default),
                     "malicious": int((lab0 == 1).sum()), "tp": int(((lab0 == 1) & (y[tr] == 1)).sum()),
                     "malicious_10x": int((lab1 == 1).sum()), "tp_10x": int(((lab1 == 1) & (y[tr] == 1)).sum()),
                     "labels_change_10x": int((lab0 != lab1).sum()),
                     "labels_equal_converged_all_oracles": bool((lab1 == l1).all())})
    out["ds_iteration_check_variants"] = rows
    write_json(ensure_dir(ctx.ws.root / "calibration") / "label_check.json", out)
    return out


# --------------------------------------------------------------------------- synthetic (validation only)
def stage_synthetic(cfg) -> dict:
    """Build a workspace from synthetic flows.  For code validation only."""
    from .synth import make_synthetic_flows

    ws = workspace(cfg)
    s = cfg.synthetic
    flows = make_synthetic_flows(str(_abs(cfg.corpus.schedule)), s.n_benign_per_day, s.attack_scale, s.n_hosts, s.seed,
                                 days=list(cfg.corpus.days))
    stream, info = assemble_stream(flows, cfg.corpus.schedule)
    info["SYNTHETIC"] = True
    ws.save_stream(stream, info)
    rep = ws.build_corpus(stream, cfg)
    # labelled "source" data for the transfer oracle: a second synthetic set
    src = make_synthetic_flows(str(_abs(cfg.corpus.schedule)), max(2000, s.n_benign_per_day // 3), s.attack_scale * 0.6,
                               max(10, s.n_hosts // 2), s.source_seed, days=list(cfg.corpus.days)[:6])
    sched = load_schedule(cfg.corpus.schedule)
    lab = label_flows(src, sched, -4)
    keep = lab["label"].to_numpy() >= 0
    S = statistical_features(src.loc[keep]).to_numpy(dtype=np.float32)
    orc = TransferOracle(hi=cfg.oracles.transfer.hi, lo=cfg.oracles.transfer.lo, columns=list(STAT_FEATURES))
    orc.fit_source(S, lab["label"].to_numpy()[keep].astype(int), n_estimators=40, max_depth=10, n_jobs=int(cfg.compute.n_jobs))
    joblib.dump(orc, ensure_dir(ws.root / "transfer") / "transfer_rf.joblib")
    write_json(ws.root / "transfer" / "transfer_info.json", {**orc.info, "SYNTHETIC": True})
    # the same synthetic set also plays the role of the second corpus of the cross-dataset experiment
    src_root = _abs(cfg.transfer.work_dir)
    if src_root.resolve() != ws.root.resolve():
        ws2 = Workspace(src_root)
        stream2, info2 = assemble_stream(src, cfg.corpus.schedule)
        info2["SYNTHETIC"] = True
        ws2.save_stream(stream2, info2)
        rep2 = ws2.build_corpus(stream2, cfg)
        rep2["SYNTHETIC"] = True
        write_json(ws2.corpus_dir / "corpus_info.json", rep2)
    rep["SYNTHETIC"] = True
    rep["environment"] = environment_record()
    write_json(ws.corpus_dir / "corpus_info.json", rep)
    return rep

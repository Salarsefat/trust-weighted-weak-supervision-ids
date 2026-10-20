"""Diagnostics of the labelling rules added after the results (docs/ORACLES_EXTENDED.md, 2026-10-12).

They answer questions that the first report left open. Except ``diag_selection``, they use oracle votes alone: no
detector is trained; no setting is changed, and nothing here feeds back into any other experiment.

``diag_oracles`` (experiment of the extended configuration)
    Under the forward-in-time protocol: which oracle votes malicious on the flows of each attack in each subset,
    the coverage of the pseudo-labels, and how the botnet flows of the test days are covered (flagged by the
    repeated-connection oracle, by the regularity oracle only, ...).  The thresholds are fitted on the training days,
    exactly as in experiment ``protocols``; the repeated-connection threshold is written next to that of ``protocols``.

``diag_selection`` (experiment of the extended configuration)
    The selected configuration again, and with the size d' of the mutual-information filter and the number F of
    folds of the feature selection changed one at a time (the two settings of Step 5 that the ablation study keeps
    fixed), on the ablation seeds, with the hyperparameters of the search result in the work directory.  Unlike the
    other diagnostics it trains detectors; its reference row is run in the same environment.

``spacing`` (command line, on the records of the attacker-controlled endpoints of the stream)
    The votes of the repeated-connection and regularity oracles on the flows that the attacker-controlled endpoints
    send, before and after the connection spacing of experiment ``adversarial`` (the same transformation,
    ``extra._transform_stream``), with the threshold of the selected configuration.

``rate_threshold`` (command line, on the corpus of the second dataset)
    The votes of the repeated-connection oracle on the flows of selected attacks for a list of thresholds, and the
    distribution of the quantity it thresholds (connections of the pair in the last 60 s) on those flows.

Usage of the command-line parts (each writes one JSON file with the SHA-256 of its inputs):

    python -m twws.experiments.diagnostics substream --config configs/paper_reduced_ext.yaml \
        --stream WORK/corpus/stream.parquet --out SUB.parquet --manifest diag_substream.json
    python -m twws.experiments.diagnostics spacing --config configs/paper_reduced_ext.yaml \
        --substream SUB.parquet --threshold 29.1 --out diag_spacing.json
    python -m twws.experiments.diagnostics rate_threshold --corpus-dir DIR --split-seed 42 --quantile 0.999 \
        --attacks FTP-Patator,SSH-Patator --thresholds 29.1 --out diag_rate_threshold_ids2017.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..corpus import TEST, TRAIN, UNUSED, VAL_A, VAL_B, make_split
from ..oracles import ABS, BEN, MAL, BeaconOracle, RateOracle
from ..utils import get_logger, sha256_file, write_json
from ..weak import aggregate_votes
from .common import Ctx, header

log = get_logger("twws.diag")

SUBSETS = {"train": (TRAIN,), "validation": (VAL_A, VAL_B), "test": (TEST,)}


def _shares(votes: np.ndarray, mask: np.ndarray) -> dict:
    v = votes[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    return {"n": n, "malicious": float((v == MAL).mean()), "benign": float((v == BEN).mean()),
            "abstain": float((v == ABS).mean())}


def _regularity_inputs(aux: pd.DataFrame, mask: np.ndarray, beacon: BeaconOracle) -> dict:
    """Inputs of the regularity rule on the flows of ``mask``: which of its conditions the flows meet.

    ``with_interval``: share of the flows for which the pair has a median inter-arrival time; among those,
    ``iat_med_below_min`` is the share whose median is below ``iat_min`` and ``disp_at_most_max`` the share whose
    dispersion is at most ``disp_max``; ``n1h_below_min_conn``: share of all flows of ``mask`` whose pair has fewer
    than ``min_conn`` connections in the last hour."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    med = aux["aux_pair_iat_med"].to_numpy(dtype=np.float64)[mask]
    disp = aux["aux_pair_iat_disp"].to_numpy(dtype=np.float64)[mask]
    n1h = aux["aux_pair_n1h"].to_numpy(dtype=np.float64)[mask]
    has = np.isfinite(med)
    out = {"n": n, "with_interval": float(has.mean()),
           "n1h_below_min_conn": float(np.mean(~(np.isfinite(n1h) & (n1h >= beacon.min_conn))))}
    if has.any():
        out["iat_med_below_min"] = float(np.mean(med[has] < beacon.iat_min))
        out["disp_at_most_max"] = float(np.mean(disp[has] <= beacon.disp_max))
        out["iat_med_quantiles"] = {str(q): float(np.quantile(med[has], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return out


def exp_diag_oracles(ctx: Ctx) -> dict:
    """Oracle coverage per attack and subset under the forward-in-time protocol (see the module docstring)."""
    cfg = ctx.cfg
    fw = cfg.experiments.get("forward")
    if not fw:
        raise ValueError("experiments.forward is not configured")
    forward = tuple((k, tuple(fw[k])) for k in ("train", "val", "test"))
    o = ctx.base_options(protocol="forward", temporal="global", forward=forward)
    meta = ctx.ws.meta
    split = make_split(meta, o.protocol, o.split_seed, {k: list(v) for k, v in forward})
    A = {"split": split}
    use = [n for n in o.oracles if n in ctx.prep.available_oracles()]
    t0 = time.perf_counter()
    res_v = {n: ctx.prep._oracle_votes(n, o, A) for n in use}
    V = np.column_stack([res_v[n]["votes"] for n in use]).astype(np.int8)
    tr = split == TRAIN
    agg = aggregate_votes(V, o.aggregation, fit_mask=tr, names=use, group_jaccard=o.group_jaccard)
    lab = np.asarray(agg["label"])
    y = (meta["label"].to_numpy() == 1)
    fam = meta["family"].astype(str).to_numpy()
    att = meta["attack"].astype(str).to_numpy() if "attack" in meta.columns else fam
    out = {**header(ctx, "diag_oracles"), "protocol": "forward", "forward_days": {k: list(v) for k, v in forward},
           "oracles": use, "aggregation": o.aggregation,
           "describe": {n: res_v[n]["describe"] for n in use}, "subsets": {}}
    for sname, codes in SUBSETS.items():
        sm = np.isin(split, codes)
        rows = {}
        for a in sorted(set(att[sm & y])):
            m = sm & y & (att == a)
            rows[a] = {"family": str(fam[m][0]), "n": int(m.sum()),
                       "pseudo_label_malicious": float((lab[m] == 1).mean()),
                       "oracles": {n: _shares(res_v[n]["votes"], m) for n in use}}
        ben = sm & ~y
        rows["Benign"] = {"family": "Benign", "n": int(ben.sum()),
                          "pseudo_label_malicious": float((lab[ben] == 1).mean()) if ben.any() else None,
                          "oracles": {n: _shares(res_v[n]["votes"], ben) for n in use}}
        out["subsets"][sname] = rows
    # how the botnet flows of the test days are covered: which of the two repetition oracles flags them
    te_bot = (split == TEST) & y & (fam == "Bot")
    if te_bot.any() and "rate" in use and "beacon" in use:
        r = res_v["rate"]["votes"][te_bot] == MAL
        b = res_v["beacon"]["votes"][te_bot] == MAL
        pl = lab[te_bot] == 1
        others = [n for n in use if n not in ("rate", "beacon")]
        o_any = np.zeros(int(te_bot.sum()), dtype=bool)
        for n in others:
            o_any |= res_v[n]["votes"][te_bot] == MAL
        out["test_botnet"] = {
            "n": int(te_bot.sum()),
            "rate_malicious": float(r.mean()), "beacon_malicious": float(b.mean()),
            "rate_only": float((r & ~b).mean()), "beacon_only": float((b & ~r).mean()), "both": float((r & b).mean()),
            "neither": float((~r & ~b).mean()), "other_oracles_malicious": float(o_any.mean()),
            "pseudo_label_malicious": float(pl.mean()),
            "pseudo_label_given_rate": float(pl[r].mean()) if r.any() else None,
            "pseudo_label_given_beacon_only": float(pl[b & ~r].mean()) if (b & ~r).any() else None,
            "pseudo_label_given_neither": float(pl[~r & ~b].mean()) if (~r & ~b).any() else None,
        }
    out["seconds"] = round(time.perf_counter() - t0, 1)
    ctx.ws.save_result("diag_oracles", out)
    log.info("oracle coverage under the forward protocol in %.0f s", out["seconds"])
    return out


# --------------------------------------------------------------------------- settings of the feature selection
def exp_diag_selection(ctx: Ctx) -> dict:
    """Sensitivity of the feature selection (Step 5) to the two settings that the ablation study keeps fixed: the size
    d' of the mutual-information filter and the number F of folds.  Computed after the results; the reference row is
    the selected configuration run again in the same environment, so that no difference mixes two machines.  Only the
    selection changes: oracles, label model, hyperparameters (from the search result of the work directory) and seeds
    (the ablation seeds) are those of the selected configuration."""
    from .common import run_seeds, summarise
    cfg = ctx.cfg
    params = ctx.best_params()
    seeds = [int(x) for x in cfg.experiments.ablation_seeds]
    d0, f0 = int(cfg.selection.mi_top), int(cfg.selection.n_folds)
    variants = [("reference (selected configuration)", {}),
                (f"mutual-information filter d' = {d0 - 20}", {"mi_top": d0 - 20}),
                (f"mutual-information filter d' = {d0 + 20}", {"mi_top": d0 + 20}),
                ("F = 3 folds (as in the first version)", {"n_folds": 3}),
                ("F = 10 folds", {"n_folds": 10})]
    name = "diag_selection"
    res = (ctx.ws.load_result(name) if ctx.ws.has_result(name)
           else {**header(ctx, name), "reference": {"mi_top": d0, "n_folds": f0, "k": int(cfg.selection.k),
                                                     "theta": float(cfg.selection.theta)}, "variants": []})
    done = {v["name"] for v in res["variants"] if "summary" in v}
    for vname, changes in variants:
        if vname in done:
            continue
        res["variants"] = [e for e in res["variants"] if e["name"] != vname]
        t0 = time.perf_counter()
        entry = {"name": vname, "prep_changes": changes}
        try:
            prep = ctx.prep.prepare(ctx.base_options(**changes))
            runs, summ = run_seeds(prep, params, seeds, cfg, ctx.n_jobs)
            entry.update({"summary": summ, "n_features": int(prep.Xs.shape[1]), "selected": list(prep.sel_names),
                          "runs": [{k: r[k] for k in ("seed", "precision", "recall", "f1", "pr_auc", "auc_roc")
                                    if k in r} for r in runs],
                          "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as exc:                  # noqa: BLE001 - recorded, the other variants still run
            log.exception("selection variant '%s' failed", vname)
            entry["status"] = f"failed: {exc}"
        res["variants"].append(entry)
        res["seeds"] = seeds
        ctx.ws.save_result(name, res)             # checkpoint after every variant
        log.info("selection variant '%s' done in %.0f s", vname, time.perf_counter() - t0)
    return res


# --------------------------------------------------------------------------- connection spacing
SUBSTREAM_COLUMNS = ["conn", "dir", "ts", "t_first", "t_last", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
                     "day", "label", "family", "attack"]


def extract_substream(cfg, stream: Path, out: Path) -> dict:
    """Input of ``diag_spacing``: the records of the stream whose source or destination is one of the
    attacker-controlled endpoints of any capture day (``extra._controlled_endpoints``), with the columns that the
    pair statistics and the spacing transformation read, in the order of the stream.  The stream is read and written
    one row group at a time, so that the step runs in little memory (it was run on the author's computer, where the
    stream is kept)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from ..labels import load_schedule
    from .extra import _controlled_endpoints

    ips = sorted(set().union(*_controlled_endpoints(load_schedule(cfg.corpus.schedule)).values()))
    vs = pa.array(ips)
    f = pq.ParquetFile(stream)
    writer, n = None, 0
    try:
        for i in range(f.num_row_groups):
            tab = f.read_row_group(i, columns=SUBSTREAM_COLUMNS)
            keep = pc.or_(pc.is_in(tab.column("src_ip"), value_set=vs), pc.is_in(tab.column("dst_ip"), value_set=vs))
            tab = tab.filter(keep)
            if writer is None:
                writer = pq.ParquetWriter(out, tab.schema)
            writer.write_table(tab)
            n += tab.num_rows
    finally:
        if writer is not None:
            writer.close()
    return {"experiment": "diag_substream", "stream": str(stream), "stream_sha256": sha256_file(stream),
            "stream_records": int(f.metadata.num_rows), "endpoints": ips, "records": int(n),
            "out": str(out), "out_sha256": sha256_file(out)}


def diag_spacing(cfg, substream: Path, threshold: float, levels: list[float]) -> dict:
    """Votes of the two repetition oracles on the flows of attacker-controlled endpoints, with and without spacing.

    ``substream``: the records of the stream whose source or destination is an attacker-controlled endpoint (every
    pair that contains a connection of such an endpoint is complete in it, because a pair is keyed by its
    originator).  The controlled flows are those of experiment ``adversarial``: malicious records sent by a
    controlled endpoint on its capture day."""
    from ..labels import load_schedule
    from ..pairs import pair_statistics
    from .extra import _controlled_endpoints, _transform_stream

    t0 = time.perf_counter()
    s = pd.read_parquet(substream)
    # columns that the transformation reads but that spacing does not change
    for c, v in (("pkts", 1), ("bytes", 0), ("conn_duration", 0.0)):
        if c not in s.columns:
            s[c] = v
    sched = load_schedule(cfg.corpus.schedule)
    ctrl_sets = _controlled_endpoints(sched)
    lab = s["label"].to_numpy()
    ctrl = np.zeros(len(s), dtype=bool)
    for d, ips in ctrl_sets.items():
        m = (s["day"].astype(str) == d).to_numpy() & (lab == 1)
        if m.any() and ips:
            ctrl |= m & s["src_ip"].astype(str).isin(list(ips)).to_numpy()
    fam = s["family"].astype(str).to_numpy()
    rate = RateOracle(threshold=float(threshold), guard=float(cfg.get_path("oracles.rate.guard", 0.05)))
    beacon = BeaconOracle(min_conn=int(cfg.get_path("oracles.beacon.min_conn", 10)),
                          disp_max=float(cfg.get_path("oracles.beacon.disp_max", 0.1)),
                          iat_min=float(cfg.get_path("oracles.beacon.iat_min", 1.0)))
    out = {"experiment": "diag_spacing", "substream": str(substream), "substream_sha256": sha256_file(substream),
           "records": int(len(s)), "controlled_records": int(ctrl.sum()), "threshold": float(threshold),
           "guard": rate.guard, "levels": [], "families": sorted(set(fam[ctrl]))}
    for level in [1.0] + [float(x) for x in levels]:
        st = s if level == 1.0 else _transform_stream(s, ctrl, "throttle", level)
        ps = pair_statistics(st)
        pos = pd.Index(ps["conn"].to_numpy()).get_indexer(st["conn"].to_numpy())
        aux = ps.iloc[pos].reset_index(drop=True)
        rv, bv = rate.vote(aux), beacon.vote(aux)
        row = {"level": level, "families": {}}
        for f in out["families"]:
            m = ctrl & (fam == f)
            row["families"][f] = {"n": int(m.sum()), "rate": _shares(rv, m), "beacon": _shares(bv, m),
                                  "either_malicious": float(((rv == MAL) | (bv == MAL))[m].mean()),
                                  "regularity_inputs": _regularity_inputs(aux, m, beacon)}
        m = ctrl
        row["all"] = {"n": int(m.sum()), "rate": _shares(rv, m), "beacon": _shares(bv, m),
                      "either_malicious": float(((rv == MAL) | (bv == MAL))[m].mean())}
        out["levels"].append(row)
        log.info("spacing x%g: rate malicious %.4f, beacon malicious %.4f", level, row["all"]["rate"]["malicious"],
                 row["all"]["beacon"]["malicious"])
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# --------------------------------------------------------------------------- threshold of the repeated-connection oracle
def diag_rate_threshold(corpus_dir: Path, split_seed: int, quantile: float, guard: float, attacks: list[str],
                        thresholds: list[float]) -> dict:
    """Votes of the repeated-connection oracle on the flows of ``attacks`` for several thresholds (second dataset)."""
    t0 = time.perf_counter()
    meta = pd.read_parquet(corpus_dir / "meta.parquet")
    pairs = pd.read_parquet(corpus_dir / "aux_pairs.parquet")
    if len(pairs) != len(meta):
        raise ValueError("aux_pairs.parquet and meta.parquet differ in length")
    split = make_split(meta, "random", int(split_seed))
    own = RateOracle.pair_threshold(pairs["aux_pair_id"].to_numpy(), pairs["aux_pair_n60"].to_numpy(),
                                    split == TRAIN, float(quantile))
    att = meta["attack"].astype(str).to_numpy()
    n60 = pairs["aux_pair_n60"].to_numpy(dtype=np.float64)
    # which pairs set the threshold: the pairs whose peak over the training rows exceeds it, and how many of them
    # contain a malicious training flow
    ok = (split == TRAIN) & (pairs["aux_pair_id"].to_numpy() >= 0) & np.isfinite(n60)
    g = pd.DataFrame({"pid": pairs["aux_pair_id"].to_numpy()[ok], "n60": n60[ok],
                      "mal": meta["label"].to_numpy()[ok] == 1}).groupby("pid").agg(peak=("n60", "max"), mal=("mal", "max"))
    top = g[g["peak"] > own]
    out = {"experiment": "diag_rate_threshold", "corpus_dir": str(corpus_dir),
           "sha256": {f: sha256_file(corpus_dir / f) for f in ("meta.parquet", "aux_pairs.parquet")},
           "split_seed": int(split_seed), "quantile": float(quantile), "guard": float(guard),
           "threshold_own": float(own), "pairs_train": int(len(g)), "pairs_above_own": int(len(top)),
           "pairs_above_own_with_attack": int(top["mal"].sum()), "attacks": {}}
    for a in attacks:
        m = (att == a) & (split == TRAIN)
        x = n60[m]
        fin = x[np.isfinite(x)]
        rec = {"n_train": int(m.sum()), "tcp_share": float(np.isfinite(x).mean()) if len(x) else None,
               "n60_quantiles": ({str(q): float(np.quantile(fin, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
                                 if len(fin) else {}),
               "n60_max": float(fin.max()) if len(fin) else None, "votes": {}}
        for thr in [float(own)] + [float(t) for t in thresholds]:
            v = RateOracle(threshold=thr, guard=float(guard)).vote(pairs.loc[m, ["aux_pair_n60"]])
            rec["votes"][f"{thr:g}"] = _shares(v, np.ones(len(v), dtype=bool))
        out["attacks"][a] = rec
    # the cost of a threshold: share of the benign training flows that it labels malicious
    ben = (meta["label"].to_numpy() != 1) & (split == TRAIN)
    out["benign_train"] = {"n": int(ben.sum()), "votes": {}}
    for thr in [float(own)] + [float(t) for t in thresholds]:
        v = RateOracle(threshold=thr, guard=float(guard)).vote(pairs.loc[ben, ["aux_pair_n60"]])
        out["benign_train"]["votes"][f"{thr:g}"] = _shares(v, np.ones(len(v), dtype=bool))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    from ..config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("substream")
    a0.add_argument("--config", required=True)
    a0.add_argument("--stream", required=True)
    a0.add_argument("--out", required=True)
    a0.add_argument("--manifest", required=True)
    a1 = sub.add_parser("spacing")
    a1.add_argument("--config", required=True)
    a1.add_argument("--substream", required=True)
    a1.add_argument("--threshold", type=float, required=True)
    a1.add_argument("--levels", default="")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("rate_threshold")
    a2.add_argument("--corpus-dir", required=True)
    a2.add_argument("--split-seed", type=int, default=42)
    a2.add_argument("--quantile", type=float, default=0.999)
    a2.add_argument("--guard", type=float, default=0.05)
    a2.add_argument("--attacks", required=True)
    a2.add_argument("--thresholds", default="")
    a2.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "substream":
        res = extract_substream(load_config(args.config), Path(args.stream), Path(args.out))
        write_json(Path(args.manifest), res)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))}, indent=1))
        return
    if args.cmd == "spacing":
        cfg = load_config(args.config)
        levels = [float(x) for x in args.levels.split(",") if x] or [float(x) for x in cfg.experiments.adversarial.throttle]
        res = diag_spacing(cfg, Path(args.substream), args.threshold, levels)
    else:
        res = diag_rate_threshold(Path(args.corpus_dir), args.split_seed, args.quantile, args.guard,
                                  [x for x in args.attacks.split(",") if x],
                                  [float(x) for x in args.thresholds.split(",") if x])
    write_json(Path(args.out), res)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))}, indent=1))


if __name__ == "__main__":
    main()

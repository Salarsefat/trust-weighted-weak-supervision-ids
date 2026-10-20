"""Command-line entry point.

    python -m twws.cli fetch      --config configs/paper.yaml [--list-only] [--local-root DIR] [--days D1,D2]
    python -m twws.cli extract    --config configs/paper.yaml
    python -m twws.cli ingest     --config configs/paper.yaml [--local-root DIR] [--delete-pcaps]   (fetch+extract per day)
    python -m twws.cli corpus     --config configs/paper.yaml
    python -m twws.cli pairs      --config configs/paper.yaml        (pair statistics of an existing corpus)
    python -m twws.cli transfer   --config configs/paper.yaml
    python -m twws.cli synthetic  --config configs/smoke.yaml        (code validation only)
    python -m twws.cli run        --config configs/paper.yaml main oracles ...
    python -m twws.cli run-all    --config configs/paper.yaml
    python -m twws.cli report     --config configs/paper_reduced_ext.yaml --out work/report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from .config import apply_dotted_overrides, load_config
from .utils import get_logger, write_json

log = get_logger("twws.cli")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="twws", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "extract", "ingest", "corpus", "pairs", "transfer", "oracle-check", "label-check",
                                        "synthetic", "run", "run-all", "report"])
    ap.add_argument("names", nargs="*", help="experiment names for 'run'")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a configuration value")
    ap.add_argument("--list-only", action="store_true", help="fetch: list and select members without downloading")
    ap.add_argument("--local-root", default=None,
                    help="fetch/ingest: folder with the daily archives (<root>/<day>/pcap.zip) or with extracted captures")
    ap.add_argument("--days", default=None, help="fetch/ingest: comma-separated subset of capture days")
    ap.add_argument("--delete-pcaps", action="store_true", help="ingest: delete the captures of a day after extraction")
    ap.add_argument("--force", action="store_true", help="run: recompute results that already exist")
    ap.add_argument("--out", default=None, help="report: output directory")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = apply_dotted_overrides(load_config(args.config), args.set)

    from . import stages

    days = [d.strip() for d in args.days.split(",")] if args.days else None
    if args.command == "fetch":
        print(json.dumps(stages.stage_fetch(cfg, list_only=args.list_only, local_root=args.local_root, days=days), indent=2))
        return 0
    if args.command == "ingest":
        print(json.dumps(stages.stage_ingest(cfg, local_root=args.local_root, delete_pcaps=args.delete_pcaps, days=days),
                         indent=2, default=str))
        return 0
    if args.command == "extract":
        print(json.dumps(stages.stage_extract(cfg), indent=2, default=str))
        return 0
    if args.command == "corpus":
        rep = stages.stage_corpus(cfg)
        print(json.dumps({k: rep[k] for k in ("n_selected", "malicious_share", "selected", "available")}, indent=2))
        return 0
    if args.command == "pairs":
        print(json.dumps(stages.stage_pairs(cfg), indent=2, default=str))
        return 0
    if args.command == "transfer":
        print(json.dumps(stages.stage_transfer(cfg), indent=2, default=str))
        return 0
    if args.command == "oracle-check":
        print(json.dumps(stages.stage_oracle_check(cfg), indent=2, default=str))
        return 0
    if args.command == "label-check":
        rep = stages.stage_label_check(cfg)
        print(json.dumps({"oracles": rep["oracles"],
                          "aggregation_train": {m: {k: v for k, v in r.items() if k in
                                                    ("precision", "recall_all_malicious", "pseudo_malicious_share")}
                                                for m, r in rep["aggregation_train"].items()}}, indent=2, default=str))
        return 0
    if args.command == "synthetic":
        rep = stages.stage_synthetic(cfg)
        print(json.dumps({k: rep[k] for k in ("n_selected", "malicious_share", "selected")}, indent=2))
        return 0
    if args.command == "report":
        from .report.build import build_report

        build_report(cfg, args.out or "work/report")
        return 0

    from .experiments import ORDER, registry
    from .experiments.common import Ctx

    reg = registry()
    # these two checkpoint after every variant/model: called again, they continue where they stopped and
    # retry the variants/models that failed
    RESUMABLE = ("ablations", "baselines")
    names = ORDER if args.command == "run-all" else args.names
    unknown = [n for n in names if n not in reg]
    if unknown:
        ap.error(f"unknown experiment(s): {unknown}; available: {sorted(reg)}")
    ctx = Ctx(cfg, n_jobs=args.n_jobs)
    status = {}
    for n in names:
        if ctx.ws.has_result(n) and not args.force and n not in RESUMABLE:
            log.info("skip %s (result exists; use --force to recompute)", n)
            status[n] = "exists"
            continue
        t0 = time.perf_counter()
        try:
            reg[n](ctx)
            status[n] = f"ok ({time.perf_counter() - t0:.0f} s)"
        except Exception as exc:  # keep going with the remaining experiments
            log.error("experiment %s failed: %s\n%s", n, exc, traceback.format_exc())
            status[n] = f"FAILED: {exc}"
        write_json(ctx.ws.results_dir / "_run_status.json", status)
    print(json.dumps(status, indent=2))
    return 0 if not any(v.startswith("FAILED") for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

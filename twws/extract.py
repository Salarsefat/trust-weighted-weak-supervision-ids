"""Flow extraction: Zeek on each retained capture, then two unidirectional flows per connection."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from .config import REPO_ROOT
from .utils import ensure_dir, get_logger, release_memory, stable_id, tool_version, worker_count

log = get_logger("twws.extract")

ZEEK_SCRIPT = REPO_ROOT / "zeek" / "twws_flows.zeek"

# columns of the unidirectional flow table
FLOW_COLUMNS = [
    "uid", "dir", "ts", "t_first", "t_last", "duration", "conn_duration",
    "src_ip", "dst_ip", "src_port", "dst_port", "orig_ip", "resp_ip", "proto",
    "pkts", "bytes", "rev_pkts", "rev_bytes",
    "syn", "fin", "rst", "psh", "ack", "urg",
    "pay", "cap", "entropy", "ent_n", "conn_state", "tls", "tls_self_signed",
]


def zeek_binary(cfg=None) -> str:
    exe = (cfg.get_path("tools.zeek") if cfg is not None else None) or shutil.which("zeek")
    if exe is None:
        for cand in ("/opt/zeek/bin/zeek", "/usr/local/zeek/bin/zeek", "/usr/bin/zeek"):
            if os.path.exists(cand):
                exe = cand
                break
    if exe is None:
        raise FileNotFoundError("zeek binary not found; set tools.zeek in the configuration")
    return exe


def zeek_version(cfg=None) -> str:
    exe = zeek_binary(cfg)
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()


def run_zeek(pcap: str | os.PathLike, out_dir: str | os.PathLike, zeek: str, entropy_max_bytes: int = 4096,
             bare: bool = True, timeout: float | None = None) -> Path:
    """Run Zeek on one capture; return the path of ``twws_flows.log``.

    ``-C`` ignores checksums (captures taken on end hosts with checksum offload),
    ``-b`` starts in bare mode so that only the scripts loaded by ``twws_flows.zeek`` run.
    """
    out_dir = ensure_dir(out_dir)
    cmd = [zeek, "-C"]
    if bare:
        cmd.append("-b")
    cmd += ["-r", str(Path(pcap).resolve()), str(ZEEK_SCRIPT), f"TWWS::entropy_max_bytes={int(entropy_max_bytes)}"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, timeout=timeout)
    (out_dir / "zeek.stderr").write_text(res.stderr or "")
    (out_dir / "zeek.cmd").write_text(" ".join(cmd) + "\n")
    logp = out_dir / "twws_flows.log"
    if res.returncode != 0:
        # a fatal error ends Zeek without writing the connections that are still open: never use such a log
        raise RuntimeError(f"zeek failed on {pcap} (exit code {res.returncode}): {(res.stderr or '')[-800:]}")
    if not logp.exists():  # capture without any connection
        logp.write_text("")
    (out_dir / "zeek.seconds").write_text(f"{time.perf_counter() - t0:.3f}\n")
    return logp


def read_zeek_tsv(path: str | os.PathLike) -> pd.DataFrame:
    """Parse a Zeek ASCII (TSV) log using its ``#fields`` header."""
    fields = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                break
            if not line.startswith("#"):
                break
    if fields is None:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", comment="#", names=fields, na_values=["-", "(empty)"], keep_default_na=False,
                       low_memory=False, dtype={"uid": str, "orig_h": str, "resp_h": str, "conn_state": str,
                                                "history": str, "proto": str, "tls": str, "tls_self_signed": str})


def _zbool(s: pd.Series) -> np.ndarray:
    """Zeek booleans ('T'/'F', missing) -> float array (1, 0, NaN)."""
    out = np.full(len(s), np.nan, dtype=np.float32)
    v = s.astype("object").to_numpy()
    out[v == "T"] = 1.0
    out[v == "F"] = 0.0
    return out


def flows_from_log(log_path: str | os.PathLike) -> pd.DataFrame:
    """Split every connection of ``twws_flows.log`` into its two unidirectional flows.

    A direction that carried no packet is not a flow and is dropped.  The per-direction
    duration is ``last - first`` packet time of that direction; for protocols without
    per-packet handlers (ICMP, others) the connection duration is used for both directions.
    """
    c = read_zeek_tsv(log_path)
    if c.empty:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    num = ["ts", "duration", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "o_first", "o_last",
           "r_first", "r_last", "o_seen", "r_seen", "o_syn", "o_fin", "o_rst", "o_psh", "o_ack", "o_urg", "r_syn",
           "r_fin", "r_rst", "r_psh", "r_ack", "r_urg", "o_pay", "o_cap", "r_pay", "r_cap", "o_ent", "o_ent_n",
           "r_ent", "r_ent_n", "orig_p", "resp_p"]
    for col in num:
        c[col] = pd.to_numeric(c[col], errors="coerce") if col in c else np.nan
    for col in ["orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]:
        c[col] = c[col].fillna(0)
    # connections that Zeek summarised without per-direction counters: fall back to handler counts
    c["orig_pkts"] = np.where(c["orig_pkts"] > 0, c["orig_pkts"], c["o_seen"].fillna(0))
    c["resp_pkts"] = np.where(c["resp_pkts"] > 0, c["resp_pkts"], c["r_seen"].fillna(0))
    tls = np.nan_to_num(_zbool(c["tls"]), nan=0.0)
    selfs = _zbool(c["tls_self_signed"])
    conn_dur = c["duration"].fillna(0.0).to_numpy()

    def one(direction: int) -> pd.DataFrame:
        a, b = ("o", "r") if direction == 0 else ("r", "o")
        A, B = ("orig", "resp") if direction == 0 else ("resp", "orig")
        first, last = c[f"{a}_first"].to_numpy(), c[f"{a}_last"].to_numpy()
        dur = np.where(np.isfinite(first) & np.isfinite(last), last - first, conn_dur)
        return pd.DataFrame({
            "uid": c["uid"].to_numpy(), "dir": np.int8(direction), "ts": c["ts"].to_numpy(),
            "t_first": np.where(np.isfinite(first), first, c["ts"].to_numpy()),
            "t_last": np.where(np.isfinite(last), last, c["ts"].to_numpy() + conn_dur),
            "duration": np.maximum(dur, 0.0), "conn_duration": conn_dur,
            "src_ip": c[f"{A}_h"].to_numpy(), "dst_ip": c[f"{B}_h"].to_numpy(),
            "src_port": c[f"{A}_p"].fillna(0).to_numpy().astype(np.int32),
            "dst_port": c[f"{B}_p"].fillna(0).to_numpy().astype(np.int32),
            "orig_ip": c["orig_h"].to_numpy(), "resp_ip": c["resp_h"].to_numpy(),
            "proto": c["proto"].fillna("unknown").to_numpy(),
            "pkts": c[f"{A}_pkts"].to_numpy().astype(np.int64), "bytes": c[f"{A}_ip_bytes"].to_numpy().astype(np.int64),
            "rev_pkts": c[f"{B}_pkts"].to_numpy().astype(np.int64),
            "rev_bytes": c[f"{B}_ip_bytes"].to_numpy().astype(np.int64),
            "syn": c[f"{a}_syn"].fillna(0).to_numpy().astype(np.int32),
            "fin": c[f"{a}_fin"].fillna(0).to_numpy().astype(np.int32),
            "rst": c[f"{a}_rst"].fillna(0).to_numpy().astype(np.int32),
            "psh": c[f"{a}_psh"].fillna(0).to_numpy().astype(np.int32),
            "ack": c[f"{a}_ack"].fillna(0).to_numpy().astype(np.int32),
            "urg": c[f"{a}_urg"].fillna(0).to_numpy().astype(np.int32),
            "pay": c[f"{a}_pay"].fillna(0).to_numpy().astype(np.int64),
            "cap": c[f"{a}_cap"].fillna(0).to_numpy().astype(np.int64),
            "entropy": c[f"{a}_ent"].to_numpy().astype(np.float32),
            "ent_n": c[f"{a}_ent_n"].fillna(0).to_numpy().astype(np.int32),
            "conn_state": c["conn_state"].fillna("OTH").to_numpy(),
            "tls": tls.astype(np.int8), "tls_self_signed": selfs,
        })

    out = pd.concat([one(0), one(1)], ignore_index=True)
    out = out[out["pkts"] > 0].reset_index(drop=True)
    return out[FLOW_COLUMNS]


def _process_capture(args) -> dict:
    pcap, work_dir, out_parquet, zeek, ent, bare, keep_logs, meta = args
    t0 = time.perf_counter()
    logp = run_zeek(pcap, work_dir, zeek, entropy_max_bytes=ent, bare=bare)
    flows = flows_from_log(logp)
    for k, v in meta.items():
        flows[k] = v
    ensure_dir(Path(out_parquet).parent)
    flows.to_parquet(out_parquet, index=False)
    loaded = Path(work_dir) / "loaded_scripts.log"
    n_scripts = sum(1 for ln in open(loaded, errors="replace") if not ln.startswith("#")) if loaded.exists() else None
    if not keep_logs:
        for p in Path(work_dir).glob("*.log"):
            if p.name != "loaded_scripts.log":
                p.unlink()
    return {"capture": meta.get("capture"), "day": meta.get("day"), "flows": len(flows),
            "seconds": round(time.perf_counter() - t0, 2), "loaded_scripts": n_scripts}


def zeek_jobs(cfg, manifest: pd.DataFrame, data_root: str | os.PathLike, out_dir: str | os.PathLike) -> list[tuple]:
    """Arguments of ``_process_capture`` for every capture of the manifest that has no flow table yet."""
    out_dir = ensure_dir(out_dir)
    zeek = zeek_binary(cfg)
    ent = int(cfg.get_path("extract.entropy_max_bytes", 4096))
    bare = bool(cfg.get_path("extract.bare_mode", True))
    keep = bool(cfg.get_path("extract.keep_zeek_logs", False))
    jobs = []
    for i, m in manifest.reset_index(drop=True).iterrows():
        pcap = Path(m.local_path)
        if not pcap.is_absolute():
            pcap = Path(data_root) / pcap
        name = Path(m.member).name
        pq = out_dir / "flows" / m.day / f"{name}.parquet"
        if pq.exists():
            continue
        jobs.append((str(pcap), str(out_dir / "zeek" / m.day / name), str(pq), zeek, ent, bare, keep,
                     {"day": m.day, "capture": name, "cap_id": np.int32(stable_id(m.day, name))}))
    return jobs


def record_zeek_rows(rows: list[dict], out_dir: str | os.PathLike, zeek: str) -> pd.DataFrame:
    """Add the outcome of these Zeek runs to ``extract_report.csv`` (one row per capture)."""
    out_dir = ensure_dir(out_dir)
    rep = pd.DataFrame(rows)
    if not rep.empty:
        prev = out_dir / "extract_report.csv"
        merged = rep
        if prev.exists():               # a capture extracted again (after a repair) replaces its earlier row
            merged = pd.concat([pd.read_csv(prev), rep], ignore_index=True).drop_duplicates(["day", "capture"], keep="last")
        merged.to_csv(prev, index=False)
    (out_dir / "zeek_version.txt").write_text((tool_version(zeek) or "unknown") + "\n")
    return rep


def extract_manifest(cfg, manifest: pd.DataFrame, data_root: str | os.PathLike, out_dir: str | os.PathLike,
                     n_jobs: int | None = None) -> pd.DataFrame:
    """Run Zeek on every capture of the manifest (in parallel) and write one parquet per capture."""
    jobs = zeek_jobs(cfg, manifest, data_root, out_dir)
    rows = []
    n_jobs = n_jobs or worker_count(cfg.get_path("extract.n_jobs"))
    if jobs:
        log.info("running Zeek on %d captures with %d workers", len(jobs), n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_process_capture, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                rows.append(r)
                log.info("zeek done: %s/%s %d flows in %.0f s", r["day"], r["capture"], r["flows"], r["seconds"])
    return record_zeek_rows(rows, out_dir, zeek_binary(cfg))


CATEGORY_COLUMNS = ["src_ip", "dst_ip", "orig_ip", "resp_ip", "proto", "conn_state", "day", "capture"]


def load_flows(out_dir: str | os.PathLike, days: list[str] | None = None) -> pd.DataFrame:
    """Concatenate the per-capture flow tables (string columns as categories to save memory).

    Each table is converted to categories as it is read and the categories are unified before the
    tables are concatenated, so the string columns of the whole stream are never held as strings
    (tens of millions of flows on the full corpus).  The result equals concatenating the tables and
    then converting each string column with ``astype("category")``: categories in sorted order.
    """
    import pyarrow.parquet as pq

    root = Path(out_dir) / "flows"
    parts = []
    for p in sorted(root.glob("*/*.parquet")):
        if days and p.parent.name not in days:
            continue
        # string columns are read as dictionaries, i.e. straight into categories
        table = pq.read_table(p, read_dictionary=[c for c in CATEGORY_COLUMNS if c in pq.read_schema(p).names])
        df = table.to_pandas(self_destruct=True, split_blocks=True)
        del table
        if len(df):
            for col in CATEGORY_COLUMNS:
                if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype("category")
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=FLOW_COLUMNS + ["day", "capture", "cap_id"])
    for col in CATEGORY_COLUMNS:
        if not all(col in df.columns for df in parts):
            continue
        cats = parts[0][col].cat.categories
        for df in parts[1:]:
            cats = cats.union(df[col].cat.categories)
        cats = cats.sort_values()
        for df in parts:
            df[col] = df[col].cat.set_categories(cats)
    flows = pd.concat(parts, ignore_index=True)
    del parts
    for col in CATEGORY_COLUMNS:
        if col in flows.columns and not isinstance(flows[col].dtype, pd.CategoricalDtype):
            flows[col] = flows[col].astype("category")
    release_memory()
    return flows


def connection_ids(flows: pd.DataFrame) -> np.ndarray:
    """Dense integer id per (capture, Zeek uid); both directions of a connection share it."""
    cap = flows["cap_id"].to_numpy(dtype=np.int64) if "cap_id" in flows else np.zeros(len(flows), dtype=np.int64)
    ucode, uniq = pd.factorize(flows["uid"], sort=False)
    combined = cap * (len(uniq) + 1) + ucode
    return pd.factorize(combined, sort=False)[0].astype(np.int64)


@njit(cache=True)
def _dedup_scan(same_key, ts, cap, tol):  # pragma: no cover - compiled
    """Mark second observations among records sorted by (connection key, start time).

    ``same_key[i]`` says that record i has the key of record i-1.  The last retained record of a
    key absorbs at most one record from every other capture; a record from a capture that has
    already been seen for it starts a new connection.
    """
    n = len(ts)
    dup = np.zeros(n, dtype=np.bool_)
    seen = np.empty(64, dtype=np.int64)   # captures that have contributed to the retained connection
    n_seen = 0
    last = 0
    if n > 0:
        seen[0] = cap[0]
        n_seen = 1
    for i in range(1, n):
        is_dup = False
        if same_key[i] and ts[i] - ts[last] <= tol:
            known = False
            for j in range(n_seen):
                if seen[j] == cap[i]:
                    known = True
                    break
            if not known:
                is_dup = True
                if n_seen < 64:
                    seen[n_seen] = cap[i]
                    n_seen += 1
        if is_dup:
            dup[i] = True
        else:
            last = i
            seen[0] = cap[i]
            n_seen = 1
    return dup


def deduplicate(flows: pd.DataFrame, tol_s: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Remove connections that were recorded more than once.

    A connection between two captured hosts appears in both captures.  De-duplication is done
    per connection so that the two directions are always kept from the same capture.  Within a
    group of connections with identical protocol, originator address/port and responder
    address/port, records are scanned in order of start time: a record is a second observation
    of the last retained connection, and is removed, if it starts within ``tol_s`` seconds of
    it and comes from a capture that has not yet contributed an observation of that connection.
    A record from a capture that has already contributed one is a new connection (the same
    five-tuple reused) and is retained.  Packet and byte counters are deliberately not part of
    the key because segmentation offload makes them differ between the two vantage points.
    """
    n0 = len(flows)
    if n0 == 0:
        return flows, {"flows_before": 0, "flows_after": 0, "flows_removed": 0, "tolerance_s": tol_s}
    conn = connection_ids(flows)
    first = np.unique(conn, return_index=True)[1]            # one representative row per connection
    # only the columns of the key (the full rows of half the stream would be a copy of it)
    rep = flows[[c for c in ("dir", "src_port", "dst_port", "proto", "orig_ip", "resp_ip", "ts", "cap_id")
                 if c in flows.columns]].iloc[first]
    is_o = rep["dir"].to_numpy() == 0
    o_port = np.where(is_o, rep["src_port"].to_numpy(), rep["dst_port"].to_numpy()).astype(np.int64)
    r_port = np.where(is_o, rep["dst_port"].to_numpy(), rep["src_port"].to_numpy()).astype(np.int64)
    k_proto = pd.factorize(rep["proto"].astype(str))[0]
    k_o = pd.factorize(rep["orig_ip"].astype(str))[0]
    k_r = pd.factorize(rep["resp_ip"].astype(str))[0]
    ts = rep["ts"].to_numpy(dtype=float)
    cap = rep["cap_id"].to_numpy() if "cap_id" in rep else np.zeros(len(rep), dtype=np.int64)
    order = np.lexsort((ts, r_port, k_r, o_port, k_o, k_proto))
    same_key = np.zeros(len(rep), dtype=bool)
    o = order
    same_key[1:] = ((k_proto[o][1:] == k_proto[o][:-1]) & (k_o[o][1:] == k_o[o][:-1]) & (o_port[o][1:] == o_port[o][:-1])
                    & (k_r[o][1:] == k_r[o][:-1]) & (r_port[o][1:] == r_port[o][:-1]))
    dup_sorted = _dedup_scan(same_key, np.ascontiguousarray(ts[o]), np.ascontiguousarray(cap[o]).astype(np.int64),
                             float(tol_s))
    dup_conn = np.zeros(len(rep), dtype=bool)
    dup_conn[o[dup_sorted]] = True
    conn_of_rep = conn[first]
    drop_ids = set(conn_of_rep[dup_conn].tolist())
    keep = ~np.isin(conn, np.fromiter(drop_ids, dtype=np.int64, count=len(drop_ids))) if drop_ids else np.ones(n0, bool)
    out = flows[keep].reset_index(drop=True)
    stats = {"flows_before": int(n0), "flows_after": int(len(out)), "flows_removed": int(n0 - len(out)),
             "connections_before": int(len(rep)), "connections_removed": int(dup_conn.sum()), "tolerance_s": tol_s}
    return out, stats


def check_captures(manifest: pd.DataFrame, data_root: str | os.PathLike, extract_dir: str | os.PathLike,
                   snort_dir: str | os.PathLike) -> pd.DataFrame:
    """Check (and if necessary repair) every capture of the manifest that is on disk; see ``twws.pcapcheck``.

    Outputs computed from a capture before its repair are removed, so that Zeek and Snort run again on the
    repaired file.  The outcome per capture is kept in ``<extract_dir>/capture_checks.csv``.
    """
    import shutil

    from .pcapcheck import check_and_repair

    extract_dir, snort_dir = Path(extract_dir), Path(snort_dir)
    rows = []
    for m in manifest.reset_index(drop=True).itertuples():
        pcap = Path(m.local_path)
        if not pcap.is_absolute():
            pcap = Path(data_root) / pcap
        if not pcap.exists():
            continue
        name = Path(m.member).name
        info = check_and_repair(pcap)
        if info["repaired_now"]:
            log.warning("%s/%s: %s; repaired (%d stray region(s), %d bytes removed, %d records kept)", m.day, name,
                        info["status"], info["stray_regions"], info["removed_bytes"], info["records"])
            for out in (extract_dir / "flows" / m.day / f"{name}.parquet", snort_dir / "alerts" / m.day / f"{name}.parquet"):
                out.unlink(missing_ok=True)
            shutil.rmtree(extract_dir / "zeek" / m.day / name, ignore_errors=True)
            shutil.rmtree(snort_dir / "snort" / m.day / name, ignore_errors=True)
        rows.append({"day": m.day, "capture": name, **info})
    cur = pd.DataFrame(rows)
    if not cur.empty:
        ensure_dir(extract_dir)
        prev = extract_dir / "capture_checks.csv"
        rep = cur.drop(columns=["repaired_now"])
        if prev.exists():
            rep = pd.concat([pd.read_csv(prev), rep], ignore_index=True).drop_duplicates(["day", "capture"], keep="last")
        rep.to_csv(prev, index=False)
    return cur

"""Signature oracle: replay captures through Snort 2.9 and map alerts onto connections."""
from __future__ import annotations

import calendar
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, sha256_file, stable_id, tool_version, worker_count

log = get_logger("twws.snort")

ALERT_RE = re.compile(
    r"^(\d\d)/(\d\d)(?:/(\d\d))?-(\d\d):(\d\d):(\d\d)\.(\d+)\s+\[\*\*\]\s+\[(\d+):(\d+):(\d+)\]\s+(.*?)\s+\[\*\*\]"
    r".*?\{(\w[\w-]*)\}\s+(\S+)\s+->\s+(\S+)\s*$")


def snort_binary(cfg=None) -> str:
    exe = (cfg.get_path("tools.snort") if cfg is not None else None) or shutil.which("snort")
    if exe is None:
        raise FileNotFoundError("snort binary not found; set tools.snort in the configuration")
    return exe


def rule_classtypes(rule_files: list[str]) -> dict[int, str]:
    """``sid -> classtype`` of the text rules (generator 1) in these rule files, comment lines included
    (an alert can only come from an active rule, so disabled duplicates do no harm)."""
    out: dict[int, str] = {}
    sid_re, cls_re = re.compile(r"\bsid\s*:\s*(\d+)"), re.compile(r"\bclasstype\s*:\s*([\w-]+)")
    for rf in rule_files:
        with open(rf, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "sid" not in line:
                    continue
                ms, mc = sid_re.search(line), cls_re.search(line)
                if ms and mc and not line.lstrip().startswith("#"):
                    out[int(ms.group(1))] = mc.group(1)
                elif ms and mc:
                    out.setdefault(int(ms.group(1)), mc.group(1))
    return out


def attack_alert_mask(alerts: pd.DataFrame, classtypes: dict[int, str], attack_classes) -> np.ndarray:
    """Alerts of text rules whose class is one of ``attack_classes``; preprocessor and decoder events
    (generator other than 1) carry no class and are not counted."""
    if alerts.empty:
        return np.zeros(0, dtype=bool)
    cls = alerts["sid"].map(classtypes)
    return ((alerts["gid"].to_numpy() == 1) & cls.isin(list(attack_classes)).to_numpy())


def count_rules(path: str | os.PathLike) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.lstrip()
            if s.startswith(("alert ", "drop ", "log ", "pass ", "reject ", "sdrop ")):
                n += 1
    return n


def build_conf(rule_files: list[str], out_conf: str | os.PathLike, base_conf: str = "/etc/snort/snort.conf") -> dict:
    """Derive a Snort configuration from the distribution's snort.conf.

    * With ``rule_files``: every ``include ...rules`` line of the base file is disabled and
      replaced by the given rule files (e.g. the Snort community rule set).
    * Without: the rule files enabled in the base file are used unchanged.

    HOME_NET and EXTERNAL_NET are set to ``any`` because captures are replayed offline and
    attackers and victims share address space in the test bed; output plugins are disabled
    (alerts are written with ``-A fast``); relative includes are made absolute.
    """
    out_conf = Path(out_conf)
    ensure_dir(out_conf.parent)
    base_dir = Path(base_conf).parent
    replace = bool(rule_files)
    variables = {}
    lines, used = [], []
    with open(base_conf, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            mvar = re.match(r"^var\s+(\w+)\s+(\S+)", s)
            if mvar:
                variables[mvar.group(1)] = mvar.group(2)
            minc = re.match(r"^include\s+(\S+\.rules)\s*$", s)
            if minc:
                if replace:
                    lines.append("# [twws] " + line)
                else:
                    path = minc.group(1)
                    for k, v in variables.items():
                        path = path.replace("$" + k, v)
                    if not os.path.isabs(path):
                        path = str(base_dir / path)
                    if os.path.exists(path):
                        used.append(path)
                        lines.append(f"include {path}\n")
                    else:
                        lines.append("# [twws] missing: " + line)
            elif re.match(r"^(ip)?var\s+HOME_NET\b", s):
                lines.append("ipvar HOME_NET any\n")
            elif re.match(r"^(ip)?var\s+EXTERNAL_NET\b", s):
                lines.append("ipvar EXTERNAL_NET any\n")
            elif re.match(r"^include\s+[^$/\s]+\s*$", s):          # include relative to the base file
                lines.append(f"include {base_dir / s.split()[1]}\n")
            elif re.match(r"^output\s", s):
                lines.append("# [twws] " + line)
            elif "iis_unicode_map unicode.map" in line:
                lines.append(line.replace("iis_unicode_map unicode.map", f"iis_unicode_map {base_dir / 'unicode.map'}"))
            else:
                lines.append(line)
    if replace:
        lines.append("\n# [twws] rule files\n")
        for rf in rule_files:
            rf = str(Path(rf).resolve())
            lines.append(f"include {rf}\n")
            used.append(rf)
    out_conf.write_text("".join(lines))
    info = [{"file": rf, "sha256": sha256_file(rf), "rules": count_rules(rf)} for rf in used]
    return {"conf": str(out_conf), "base_conf": base_conf, "rule_source": "configured" if replace else "distribution default",
            "rule_files": info, "active_rules": int(sum(i["rules"] for i in info))}


def run_snort(pcap: str | os.PathLike, out_dir: str | os.PathLike, conf: str, snort: str,
              timeout: float | None = None) -> Path:
    """Replay one capture; alerts go to ``<out_dir>/alert`` in 'fast' format with UTC timestamps."""
    out_dir = ensure_dir(out_dir)
    alert = out_dir / "alert"
    alert.unlink(missing_ok=True)
    # without -q: with it Snort also suppresses the reason when it stops (e.g. "truncated dump file")
    cmd = [snort, "-c", conf, "-r", str(Path(pcap).resolve()), "-A", "fast", "-l", str(out_dir),
           "-K", "none", "-U", "-y", "-k", "none"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    (out_dir / "snort.cmd").write_text(" ".join(cmd) + "\n")
    (out_dir / "snort.stderr").write_text((res.stderr or "")[-20000:])
    (out_dir / "snort.stdout").write_text((res.stdout or "")[-20000:])
    if res.returncode != 0:
        msg = "\n".join(ln for ln in ((res.stderr or "") + "\n" + (res.stdout or "")).splitlines()
                        if re.search(r"ERROR|FATAL|Fatal|truncated|Can't|failed", ln))
        raise RuntimeError(f"snort failed on {pcap} (exit code {res.returncode}): {msg[-1500:] or (res.stderr or '')[-800:]}")
    if not alert.exists():
        alert.write_text("")
    (out_dir / "snort.seconds").write_text(f"{time.perf_counter() - t0:.3f}\n")
    return alert


def _split_host_port(s: str) -> tuple[str, int]:
    if s.count(":") == 1:                      # IPv4 with port
        h, p = s.split(":")
        return h, int(p)
    if s.count(":") > 1 and not s.startswith("["):   # IPv6, possibly with a trailing :port
        h, _, p = s.rpartition(":")
        if p.isdigit() and h.count(":") >= 2 and len(p) <= 5 and "." not in p:
            # ambiguous by construction; Snort prints ports for TCP/UDP only
            return h, int(p)
        return s, 0
    return s, 0


def parse_alerts(path: str | os.PathLike, default_year: int | None = None) -> pd.DataFrame:
    """Parse a Snort 'fast' alert file into a table (ts, gid, sid, rev, msg, proto, src, sport, dst, dport)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ALERT_RE.match(line)
            if not m:
                continue
            mo, dd, yy, hh, mi, ss, frac, gid, sid, rev, msg, proto, a, b = m.groups()
            year = (2000 + int(yy)) if yy is not None else (default_year or 1970)
            ts = calendar.timegm((year, int(mo), int(dd), int(hh), int(mi), int(ss))) + float("0." + frac)
            sh, sp = _split_host_port(a)
            dh, dp = _split_host_port(b)
            rows.append((ts, int(gid), int(sid), int(rev), msg, proto.lower(), sh, sp, dh, dp))
    return pd.DataFrame(rows, columns=["ts", "gid", "sid", "rev", "msg", "proto", "src", "sport", "dst", "dport"])


def _canon_codes(*tables) -> list[np.ndarray]:
    """Direction-independent connection keys as integers, coded jointly over several tables.

    Each table is ``(proto, a, a_port, b, b_port)``.  Two rows get the same integer exactly when they have the
    same protocol (ignoring case) and the same unordered pair of endpoints; the endpoints are ordered by address
    string, then by port.  No string is built per row (the connection table of the full corpus has tens of
    millions of rows).
    """
    sizes = [len(t[1]) for t in tables]

    def joined(cols) -> pd.Series:      # one field of all tables, as strings
        parts = [(c if isinstance(c, pd.Series) else pd.Series(c)).astype(str).reset_index(drop=True) for c in cols]
        return pd.concat(parts, ignore_index=True)

    ips = pd.concat([joined([t[1] for t in tables]), joined([t[3] for t in tables])], ignore_index=True)
    ip_code = pd.factorize(ips, sort=True)[0].astype(np.int64)     # sorted: code order = address string order
    n = sum(sizes)
    ca, cb = ip_code[:n], ip_code[n:]
    ap = np.concatenate([np.asarray(t[2]).astype(np.int64) for t in tables])
    bp = np.concatenate([np.asarray(t[4]).astype(np.int64) for t in tables])
    swap = (ca > cb) | ((ca == cb) & (ap > bp))
    proto = pd.factorize(joined([t[0] for t in tables]).str.lower())[0].astype(np.int64)
    key = pd.DataFrame({"p": proto, "lo": np.where(swap, cb, ca), "lop": np.where(swap, bp, ap),
                        "hi": np.where(swap, ca, cb), "hip": np.where(swap, ap, bp)})
    key = key.groupby(["p", "lo", "lop", "hi", "hip"], sort=False).ngroup().to_numpy().astype(np.int64)
    out, start = [], 0
    for m in sizes:
        out.append(key[start:start + m])
        start += m
    return out


def alerts_to_connections(alerts: pd.DataFrame, conns: pd.DataFrame, slack_s: float = 1.0) -> pd.DataFrame:
    """Mark connections that carry at least one alert.

    ``conns`` needs conn (id), proto, orig_ip, orig_port, resp_ip, resp_port, ts, conn_duration.
    An alert belongs to a connection when the unordered endpoint pair and protocol match and
    the alert time lies in ``[ts - slack, ts + duration + slack]``.  Returns one row per
    connection with ``snort_alert`` (bool), ``snort_n`` and ``snort_sid`` (first matching sid).
    """
    out = pd.DataFrame({"conn": conns["conn"].to_numpy(), "snort_alert": False, "snort_n": 0, "snort_sid": 0})
    if alerts.empty or conns.empty:
        return out
    a = alerts.copy()
    a_key, c_key = _canon_codes((a["proto"], a["src"], a["sport"], a["dst"], a["dport"]),
                                (conns["proto"], conns["orig_ip"], conns["orig_port"], conns["resp_ip"], conns["resp_port"]))
    a["key"] = a_key
    c = pd.DataFrame({
        "conn": conns["conn"].to_numpy(),
        "key": c_key,
        "start": conns["ts"].to_numpy(dtype=float) - slack_s,
        "end": conns["ts"].to_numpy(dtype=float) + conns["conn_duration"].to_numpy(dtype=float) + slack_s,
    })
    c = c[c["key"].isin(set(a["key"].unique()))]
    if c.empty:
        return out
    a = a.sort_values("ts")[["ts", "key", "sid"]]
    a["cum"] = a.groupby("key").cumcount() + 1                       # alerts so far for this key
    left_end = c.sort_values("end")
    hit_end = pd.merge_asof(left_end, a, left_on="end", right_on="ts", by="key", direction="backward")
    left_start = c.sort_values("start")
    # alerts strictly before the start of the connection
    before = pd.merge_asof(left_start, a, left_on="start", right_on="ts", by="key", direction="backward",
                           allow_exact_matches=False)
    n_end = hit_end.set_index("conn")["cum"].fillna(0)
    n_before = before.set_index("conn")["cum"].fillna(0)
    n_in = (n_end - n_before.reindex(n_end.index)).clip(lower=0)
    first_sid = hit_end.set_index("conn")["sid"].fillna(0)
    res = pd.DataFrame({"conn": n_in.index.to_numpy(), "snort_n": n_in.to_numpy().astype(np.int64),
                        "snort_sid": first_sid.reindex(n_in.index).to_numpy().astype(np.int64)})
    res["snort_alert"] = res["snort_n"] > 0
    res.loc[~res["snort_alert"], "snort_sid"] = 0
    out = out[["conn"]].merge(res, on="conn", how="left")
    out["snort_alert"] = out["snort_alert"].fillna(False).astype(bool)
    out["snort_n"] = out["snort_n"].fillna(0).astype(np.int64)
    out["snort_sid"] = out["snort_sid"].fillna(0).astype(np.int64)
    return out


def _process(args) -> dict:
    pcap, work, conf, snort, out_parquet, meta = args
    t0 = time.perf_counter()
    alert = run_snort(pcap, work, conf, snort)
    year = int(meta["day"].rsplit("-", 1)[-1]) if meta.get("day", "")[-4:].isdigit() else None
    df = parse_alerts(alert, default_year=year)
    for k, v in meta.items():
        df[k] = v
    ensure_dir(Path(out_parquet).parent)
    df.to_parquet(out_parquet, index=False)
    return {"capture": meta.get("capture"), "day": meta.get("day"), "alerts": len(df),
            "seconds": round(time.perf_counter() - t0, 2)}


def snort_setup(cfg, out_dir: str | os.PathLike) -> dict:
    """Derived configuration, rule files and version of Snort (``info`` of the extraction record)."""
    out_dir = ensure_dir(out_dir)
    snort = snort_binary(cfg)
    rules = list(cfg.get_path("snort.rules", []) or [])
    if not rules:
        log.warning("snort.rules not set: using the rule files enabled in the distribution's snort.conf")
    info = build_conf(rules, out_dir / "snort_twws.conf", base_conf=cfg.get_path("snort.base_conf", "/etc/snort/snort.conf"))
    info["snort_version"] = tool_version(snort, ("-V",))
    info["binary"] = snort
    return info


def snort_jobs(cfg, manifest: pd.DataFrame, data_root: str | os.PathLike, out_dir: str | os.PathLike,
               info: dict) -> list[tuple]:
    """Arguments of ``_process`` for every capture of the manifest that has no alert table yet."""
    out_dir = ensure_dir(out_dir)
    jobs = []
    for i, m in manifest.reset_index(drop=True).iterrows():
        pcap = Path(m.local_path)
        if not pcap.is_absolute():
            pcap = Path(data_root) / pcap
        name = Path(m.member).name
        pq = out_dir / "alerts" / m.day / f"{name}.parquet"
        if pq.exists():
            continue
        jobs.append((str(pcap), str(out_dir / "snort" / m.day / name), info["conf"], info["binary"], str(pq),
                     {"day": m.day, "capture": name, "cap_id": np.int32(stable_id(m.day, name))}))
    return jobs


def record_snort_rows(rows: list[dict], out_dir: str | os.PathLike) -> None:
    """Add the outcome of these Snort runs to ``snort_report.csv`` (one row per capture)."""
    if not rows:
        return
    out_dir = ensure_dir(out_dir)
    rep = pd.DataFrame(rows)
    prev = out_dir / "snort_report.csv"
    if prev.exists():
        rep = pd.concat([pd.read_csv(prev), rep], ignore_index=True).drop_duplicates(["day", "capture"], keep="last")
    rep.to_csv(prev, index=False)


def snort_manifest(cfg, manifest: pd.DataFrame, data_root: str | os.PathLike, out_dir: str | os.PathLike,
                   n_jobs: int | None = None) -> dict:
    """Replay every capture of the manifest through Snort; one alert table per capture."""
    info = snort_setup(cfg, out_dir)
    jobs = snort_jobs(cfg, manifest, data_root, out_dir, info)
    n_jobs = n_jobs or worker_count(cfg.get_path("snort.n_jobs"))
    rows = []
    if jobs:
        log.info("running Snort on %d captures with %d workers (%d active rules)", len(jobs), n_jobs, info["active_rules"])
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for f in as_completed([ex.submit(_process, j) for j in jobs]):
                r = f.result()
                rows.append(r)
                log.info("snort done: %s/%s %d alerts in %.0f s", r["day"], r["capture"], r["alerts"], r["seconds"])
    record_snort_rows(rows, out_dir)
    info["captures"] = rows
    return info


def load_alerts(out_dir: str | os.PathLike) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((Path(out_dir) / "alerts").glob("*/*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["ts", "gid", "sid", "rev", "msg", "proto", "src", "sport", "dst", "dport", "day",
                                     "capture", "cap_id"])
    return pd.concat(parts, ignore_index=True)

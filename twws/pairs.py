"""Pair-level connection statistics for the oracles ``rate`` and ``beacon`` (docs/ORACLES_EXTENDED.md).

For every TCP connection c with originator o, responder r, responder port p and start time t(c),
computed over the full capture stream (every record, no label):

  n60       connections of the pair (o, r, p) that started in [t(c) - 60 s, t(c)], c included
  n1h       connections of the pair that started in [t(c) - 3600 s, t(c)], c included
  iat_med   median inter-arrival time between consecutive start times of the last (at most 50)
            connections of that one-hour set (NaN with fewer than two connections)
  iat_disp  median absolute deviation of those inter-arrival times divided by ``iat_med``
            (infinite when ``iat_med`` is zero)

Connections of the pair that start at the same instant as c count as started before it, so the
values do not depend on the order of simultaneous starts.  Both direction-specific records of a
connection receive the values of the connection; records of non-TCP connections receive NaN and
the pair id -1.  The start time of a connection is the earliest ``ts`` of its records.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import njit
from .utils import get_logger

log = get_logger("twws.pairs")

PAIR_COLUMNS = ["aux_pair_id", "aux_pair_n60", "aux_pair_n1h", "aux_pair_iat_med", "aux_pair_iat_disp"]
BEACON_LAST = 50          # inter-arrival times are taken over at most this many connections
RATE_WINDOW = 60.0
BEACON_WINDOW = 3600.0
STREAM_COLUMNS = ["conn", "dir", "ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto"]


@njit(cache=True)
def _pair_kernel(pair, t, rate_window, beacon_window, last_k):  # pragma: no cover - compiled
    """``pair`` and ``t`` sorted by (pair, t).  Returns n60, n1h, iat_med, iat_disp per position."""
    n = len(pair)
    n60 = np.zeros(n, np.float64)
    n1h = np.zeros(n, np.float64)
    med = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    buf = np.empty(last_k, np.float64)
    dev = np.empty(last_k, np.float64)
    j60 = 0
    j1h = 0
    k_end = 0
    for i in range(n):
        if i == 0 or pair[i] != pair[i - 1]:
            j60 = i
            j1h = i
            k_end = i
        if k_end < i:
            k_end = i
        # simultaneous starts count as earlier: extend to the last connection of the pair with t <= t[i]
        while k_end + 1 < n and pair[k_end + 1] == pair[i] and t[k_end + 1] <= t[i]:
            k_end += 1
        while t[j60] < t[i] - rate_window:
            j60 += 1
        while t[j1h] < t[i] - beacon_window:
            j1h += 1
        n60[i] = k_end - j60 + 1
        n1h[i] = k_end - j1h + 1
        lo = k_end - last_k + 1
        if lo < j1h:
            lo = j1h
        m = k_end - lo
        if m >= 1:
            for q in range(m):
                buf[q] = t[lo + q + 1] - t[lo + q]
            md = np.median(buf[:m])
            med[i] = md
            if md > 0.0:
                for q in range(m):
                    dev[q] = abs(buf[q] - md)
                disp[i] = np.median(dev[:m]) / md
            else:
                disp[i] = np.inf
    return n60, n1h, med, disp


def _ip_codes(stream: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer codes of source and destination addresses in one shared code space."""
    s, d = stream["src_ip"], stream["dst_ip"]
    if isinstance(s.dtype, pd.CategoricalDtype) and isinstance(d.dtype, pd.CategoricalDtype):
        uniq = pd.Index(s.cat.categories.astype(str)).union(pd.Index(d.cat.categories.astype(str)))
        ms = uniq.get_indexer(s.cat.categories.astype(str))
        md = uniq.get_indexer(d.cat.categories.astype(str))
        cs, cd = s.cat.codes.to_numpy(), d.cat.codes.to_numpy()
        if (cs < 0).any() or (cd < 0).any():
            raise ValueError("missing addresses in the stream")
        return ms[cs].astype(np.int64), md[cd].astype(np.int64), int(len(uniq))
    src = s.astype(str).to_numpy()
    dst = d.astype(str).to_numpy()
    codes, uniq = pd.factorize(np.concatenate([src, dst]), sort=False)
    n = len(src)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), int(len(uniq))


def connection_table(stream: pd.DataFrame) -> pd.DataFrame:
    """One row per connection: ``conn``, pair key, start time and whether it is TCP."""
    src_c, dst_c, n_ip = _ip_codes(stream)
    conn = stream["conn"].to_numpy(dtype=np.int64)
    d = stream["dir"].to_numpy()
    is_orig = d == 0
    o = np.where(is_orig, src_c, dst_c)
    r = np.where(is_orig, dst_c, src_c)
    p = np.where(is_orig, stream["dst_port"].to_numpy(dtype=np.int64), stream["src_port"].to_numpy(dtype=np.int64))
    ts = stream["ts"].to_numpy(dtype=np.float64)
    tcp = (stream["proto"].astype(str).to_numpy() == "tcp")
    # one record per connection: the originator record when present, else the reply record
    order = np.lexsort((d, conn))
    first = np.ones(len(order), dtype=bool)
    first[1:] = conn[order][1:] != conn[order][:-1]
    rows = order[first]
    # start time: earliest ts over the records of the connection
    t_min = pd.Series(ts).groupby(conn).min()
    key = (o[rows] * np.int64(n_ip) + r[rows]) * np.int64(65536) + (p[rows] & 0xFFFF)
    out = pd.DataFrame({"conn": conn[rows], "key": key, "tcp": tcp[rows]})
    out["t"] = t_min.reindex(out["conn"].to_numpy()).to_numpy()
    return out


def pair_statistics(stream: pd.DataFrame) -> pd.DataFrame:
    """Per connection: pair id (-1 for non-TCP), n60, n1h, iat_med, iat_disp."""
    t0 = time.perf_counter()
    ct = connection_table(stream)
    tcp = ct["tcp"].to_numpy()
    pid = np.full(len(ct), -1, dtype=np.int64)
    pid[tcp] = pd.factorize(ct["key"].to_numpy()[tcp], sort=False)[0]
    idx = np.flatnonzero(tcp)
    t = ct["t"].to_numpy(dtype=np.float64)
    order = idx[np.lexsort((t[idx], pid[idx]))]
    n60 = np.full(len(ct), np.nan)
    n1h = np.full(len(ct), np.nan)
    med = np.full(len(ct), np.nan)
    disp = np.full(len(ct), np.nan)
    if len(order):
        a, b, c, e = _pair_kernel(pid[order], t[order], RATE_WINDOW, BEACON_WINDOW, BEACON_LAST)
        n60[order], n1h[order], med[order], disp[order] = a, b, c, e
    log.info("pair statistics: %d connections (%d TCP, %d pairs) in %.0f s", len(ct), len(idx),
             int(pid.max()) + 1 if len(idx) else 0, time.perf_counter() - t0)
    return pd.DataFrame({"conn": ct["conn"].to_numpy(), "aux_pair_id": pid, "aux_pair_n60": n60,
                         "aux_pair_n1h": n1h, "aux_pair_iat_med": med, "aux_pair_iat_disp": disp})


def pair_features(stream: pd.DataFrame, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Pair statistics for the stream records ``rows`` (all records if ``None``), in that order."""
    stats = pair_statistics(stream)
    conn = stream["conn"].to_numpy(dtype=np.int64)
    rows = np.arange(len(stream)) if rows is None else np.asarray(rows, dtype=np.int64)
    pos = pd.Index(stats["conn"].to_numpy()).get_indexer(conn[rows])
    if (pos < 0).any():  # pragma: no cover - every record belongs to a connection of the table
        raise RuntimeError("records without a connection in the pair table")
    out = stats.iloc[pos].drop(columns=["conn"]).reset_index(drop=True)
    out["aux_pair_id"] = out["aux_pair_id"].astype(np.int64)
    for c in PAIR_COLUMNS[1:]:
        out[c] = out[c].astype(np.float64)
    return out[PAIR_COLUMNS]

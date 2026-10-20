"""Trace-driven emulation of the quality of service under enforcement.

The detector is not in the forwarding path: it classifies finalised flow records.  It affects the traffic only through
enforcement: after a malicious verdict, the source address of the record is blocked for a hold-down period at an edge
filter.  This module replays a traffic trace (flow records with their start, duration, availability time, bytes,
packets and ground truth) against the detector's verdicts and computes what such a filter delivers.

* A verdict is emitted when the record becomes available plus the decision latency (measured in the load test).
* A malicious verdict blocks its source for ``hold`` seconds; overlapping blocks of one source are merged.
* A record whose time span overlaps a block of its source loses the same share of its bytes and packets (bytes and
  packets are taken as uniform over the duration of a record); a record without duration is dropped when it starts
  inside a block.
* To emulate a higher offered load, every time of the trace is divided by ``speed``; the decision latency and the
  hold-down period are wall-clock quantities and are not compressed.

Everything here is deterministic; the inputs are measured (trace, verdicts, latency).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- blocks
def merged_blocks(src: np.ndarray, t_verdict: np.ndarray, hold: float):
    """Union of the hold-down intervals [t, t + hold) of every source.

    ``src`` (non-negative integers) and ``t_verdict`` (non-negative seconds) describe the records with a malicious
    verdict.  Returns (source, start, end, blocked time of the same source before the interval), sorted by source and
    start, with non-overlapping intervals per source.
    """
    src = np.asarray(src, dtype=np.int64)
    t = np.asarray(t_verdict, dtype=np.float64)
    if len(src) == 0:
        e = np.zeros(0)
        return np.zeros(0, dtype=np.int64), e, e.copy(), e.copy()
    o = np.lexsort((t, src))
    s, a = src[o], t[o]
    b = a + float(hold)
    new_src = np.r_[True, s[1:] != s[:-1]]
    seg = np.cumsum(new_src) - 1
    big = float(b.max()) + 1.0                      # larger than any end time: keys of a later source are larger
    runmax = np.maximum.accumulate(b + seg * big) - seg * big
    prev = np.r_[-np.inf, runmax[:-1]]
    start_new = new_src | (a > prev)                # a gap to everything before in the same source opens an interval
    first = np.flatnonzero(start_new)
    ms, ma = s[first], a[first]
    mb = np.maximum.reduceat(b, first)
    length = mb - ma
    cum_all = np.cumsum(length) - length            # blocked time before the interval, over all sources
    first_src = np.r_[True, ms[1:] != ms[:-1]]
    base = cum_all[first_src]
    cum = cum_all - base[np.cumsum(first_src) - 1]
    return ms, ma, mb, cum


def _locate(blocks, src_q: np.ndarray, t_q: np.ndarray, big: float):
    """Index of the last interval of the same source that starts at or before t (or -1)."""
    ms, ma, _, _ = blocks
    if len(ma) == 0:
        return np.full(len(t_q), -1, dtype=np.int64)
    key_iv = ms.astype(np.float64) * big + ma
    key_q = np.asarray(src_q, dtype=np.float64) * big + np.asarray(t_q, dtype=np.float64)
    k = np.searchsorted(key_iv, key_q, side="right") - 1
    ok = (k >= 0) & (ms[np.clip(k, 0, None)] == np.asarray(src_q))
    return np.where(ok, k, -1)


def blocked_until(blocks, src_q: np.ndarray, t_q: np.ndarray, big: float) -> np.ndarray:
    """Total blocked time of the source of every query before time t."""
    ms, ma, mb, cum = blocks
    if len(ma) == 0:
        return np.zeros(len(t_q))
    k = _locate(blocks, src_q, t_q, big)
    kk = np.clip(k, 0, len(ma) - 1)
    within = np.clip(np.asarray(t_q) - ma[kk], 0.0, mb[kk] - ma[kk])
    return np.where(k >= 0, cum[kk] + within, 0.0)


def dropped_share(blocks, src: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Share of each record [t0, t1] that falls into a block of its source (0 or 1 for records without duration)."""
    t0 = np.asarray(t0, dtype=np.float64)
    t1 = np.maximum(np.asarray(t1, dtype=np.float64), t0)
    if len(blocks[1]) == 0:
        return np.zeros(len(t0))
    big = float(max(t1.max(initial=0.0), blocks[2].max())) + 2.0
    dur = t1 - t0
    over = blocked_until(blocks, src, t1, big) - blocked_until(blocks, src, t0, big)
    frac = np.divide(over, dur, out=np.zeros_like(dur), where=dur > 0)
    point = dur <= 0
    if point.any():
        k = _locate(blocks, src[point], t0[point], big)
        kk = np.clip(k, 0, len(blocks[1]) - 1)
        inside = (k >= 0) & (t0[point] >= blocks[1][kk]) & (t0[point] < blocks[2][kk])
        frac[point] = inside.astype(np.float64)
    return np.clip(frac, 0.0, 1.0)


# --------------------------------------------------------------------------- one-second accounting
def spread_over_bins(t0: np.ndarray, t1: np.ndarray, amount: np.ndarray, n_bins: int, bin_s: float = 1.0) -> np.ndarray:
    """Distribute ``amount`` of every record uniformly over [t0, t1] into bins of ``bin_s`` seconds."""
    t0 = np.asarray(t0, dtype=np.float64) / bin_s
    t1 = np.maximum(np.asarray(t1, dtype=np.float64) / bin_s, t0)
    amount = np.asarray(amount, dtype=np.float64)
    b0 = np.clip(np.floor(t0).astype(np.int64), 0, n_bins - 1)
    b1 = np.clip(np.floor(t1).astype(np.int64), 0, n_bins - 1)
    out = np.zeros(n_bins, dtype=np.float64)
    same = b0 == b1
    out += np.bincount(b0[same], weights=amount[same], minlength=n_bins)[:n_bins]
    m = ~same
    if m.any():
        dur = t1[m] - t0[m]
        rate = amount[m] / dur                       # per bin width
        first = (b0[m] + 1 - t0[m]) * rate
        last = (t1[m] - b1[m]) * rate
        out += np.bincount(b0[m], weights=first, minlength=n_bins)[:n_bins]
        out += np.bincount(b1[m], weights=last, minlength=n_bins)[:n_bins]
        diff = np.zeros(n_bins + 1, dtype=np.float64)
        np.add.at(diff, b0[m] + 1, rate)
        np.add.at(diff, b1[m], -rate)
        out += np.cumsum(diff)[:n_bins]
    return out


# --------------------------------------------------------------------------- one emulation
def emulate(days: list[dict], *, speed: float, latency_s: float, hold_s: float, attack: bool,
            sla_tolerance: float = 0.01, bin_s: float = 1.0) -> dict:
    """Emulate the filter over all days of the trace.

    ``days``: one dict per capture day with numpy arrays ``src`` (int codes), ``ts``, ``t_avail``, ``duration``,
    ``bytes``, ``pkts``, ``label`` (0 benign, 1 malicious, other values unknown) and ``pred`` (verdict).  With
    ``attack=False`` only the benign records are replayed.
    """
    tot = {"b_off": 0.0, "b_del": 0.0, "p_off": 0.0, "p_drop": 0.0, "n_ben": 0, "n_ben_hit": 0, "n_ben_lost": 0,
           "a_off": 0.0, "a_del": 0.0, "n_att": 0, "n_att_hit": 0, "bins": 0, "bins_viol": 0, "time": 0.0,
           "src_ben": 0, "src_ben_hit": 0}
    for d in days:
        lab = d["label"]
        keep = (lab == 0) | ((lab == 1) if attack else np.zeros(len(lab), dtype=bool))
        if not keep.any():
            continue
        src = d["src"][keep]
        T0 = float(np.min(d["ts"][keep]))
        ts = (d["ts"][keep] - T0) / speed
        te = ts + np.maximum(d["duration"][keep], 0.0) / speed
        av = np.maximum((d["t_avail"][keep] - T0) / speed, te)
        pred = d["pred"][keep].astype(bool)
        y = lab[keep]
        span = float(max(av.max(), te.max())) + latency_s + hold_s
        blocks = merged_blocks(src[pred], av[pred] + latency_s, hold_s)
        frac = dropped_share(blocks, src, ts, te)
        by = d["bytes"][keep].astype(np.float64)
        pk = d["pkts"][keep].astype(np.float64)
        ben, att = y == 0, y == 1
        tot["b_off"] += by[ben].sum()
        tot["b_del"] += (by[ben] * (1.0 - frac[ben])).sum()
        tot["p_off"] += pk[ben].sum()
        tot["p_drop"] += (pk[ben] * frac[ben]).sum()
        tot["n_ben"] += int(ben.sum())
        tot["n_ben_hit"] += int((frac[ben] > 0).sum())
        tot["n_ben_lost"] += int((frac[ben] >= 1.0).sum())
        tot["a_off"] += by[att].sum()
        tot["a_del"] += (by[att] * (1.0 - frac[att])).sum()
        tot["n_att"] += int(att.sum())
        tot["n_att_hit"] += int((frac[att] > 0).sum())
        bsrc = np.unique(src[ben])
        tot["src_ben"] += len(bsrc)
        tot["src_ben_hit"] += len(np.unique(src[ben & (frac > 0)]))
        n_bins = int(np.ceil(span / bin_s)) + 1
        off = spread_over_bins(ts[ben], te[ben], by[ben], n_bins, bin_s)
        dlv = spread_over_bins(ts[ben], te[ben], by[ben] * (1.0 - frac[ben]), n_bins, bin_s)
        used = off > 0
        tot["bins"] += int(used.sum())
        tot["bins_viol"] += int((dlv[used] < (1.0 - sla_tolerance) * off[used]).sum())
        tot["time"] += span
    t = max(tot["time"], 1e-9)
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    return {
        "speed": float(speed), "hold_s": float(hold_s), "latency_ms": 1e3 * float(latency_s), "attack": bool(attack),
        "offered_benign_mbps": 8e-6 * tot["b_off"] / t, "goodput_mbps": 8e-6 * tot["b_del"] / t,
        "throughput_preservation": div(tot["b_del"], tot["b_off"]),
        "availability": 1.0 - div(tot["n_ben_hit"], tot["n_ben"]),
        "benign_records_lost_share": div(tot["n_ben_lost"], tot["n_ben"]),
        "packet_loss_ratio": div(tot["p_drop"], tot["p_off"]),
        "sla_violation_share": div(tot["bins_viol"], tot["bins"]),
        "benign_sources_hit_share": div(tot["src_ben_hit"], tot["src_ben"]),
        "attack_bytes_dropped_share": div(tot["a_off"] - tot["a_del"], tot["a_off"]) if attack else None,
        "attack_records_hit_share": div(tot["n_att_hit"], tot["n_att"]) if attack else None,
        "benign_records": tot["n_ben"], "attack_records": tot["n_att"], "seconds_emulated": t,
        "sla_bins": tot["bins"], "sla_tolerance": float(sla_tolerance),
    }


def latency_for_rate(load_rows: list[dict], rate: float) -> dict:
    """Decision latency measured in the load test for an arrival rate of ``rate`` records per second.

    The smallest measured arrival rate at or above ``rate`` that every tested worker count sustains best (lowest P95)
    is used; below the smallest measured rate, that rate is used.  Without a sustained measurement at or above
    ``rate`` the point is beyond the measured capacity.
    """
    ok = [r for r in load_rows if not r.get("saturated")]
    if not ok:
        return {"rate": rate, "within_capacity": False}
    rates = sorted({float(r["offered_rate"]) for r in ok})
    cand = [x for x in rates if x >= rate]
    if not cand:
        return {"rate": rate, "within_capacity": False}
    x = cand[0]
    best = min((r for r in ok if float(r["offered_rate"]) == x), key=lambda r: r["latency_ms"]["p95"])
    return {"rate": rate, "within_capacity": True, "measured_rate": x, "workers": int(best["workers"]),
            "p50_ms": float(best["latency_ms"]["p50"]), "p95_ms": float(best["latency_ms"]["p95"]),
            "p99_ms": float(best["latency_ms"]["p99"])}

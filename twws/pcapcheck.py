"""Structural check and repair of capture files before Zeek and Snort read them.

Some captures of CSE-CIC-IDS2018 are damaged in the archives as distributed (the ZIP CRC-32 of the
retrieved member matches the archive):

* the last packet record is incomplete (the capture process was stopped while writing it), or
* a few hundred bytes that are not a record header appear between two records (the record chain is
  broken; valid records follow again after the stray bytes).

libpcap stops at the first such place.  Snort then exits with an error, and Zeek stops with a fatal
error without writing the connections that were still open, so everything after the first break
would be lost (more than half of some captures).

``check_and_repair`` walks the record chain of a classic pcap file.  A record is accepted when its
header is plausible (time stamp within the capture period, sub-second field in range, captured length
not above the snapshot length and not above the original length).  Where the chain breaks, the
following bytes are skipped up to the next offset at which ``CHAIN`` consecutive plausible records
start (the approach of pcapfix).  The file is then rewritten from the valid records only, or cut at
the end of the last complete record when only the tail is damaged.  The outcome (records, skipped
regions and bytes, original size and SHA-256) is written next to the capture as
``<capture>.check.json``, reported by the extraction, and lets a later run recognise the repaired file.

pcapng files are only checked (block chain) and cut at the end of the last complete block.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numba import njit

PCAP_MAGICS = {                      # first four bytes -> (big endian, nanosecond time stamps)
    b"\xd4\xc3\xb2\xa1": (False, False), b"\xa1\xb2\xc3\xd4": (True, False),
    b"\x4d\x3c\xb2\xa1": (False, True), b"\xa1\xb2\x3c\x4d": (True, True),
}
PCAPNG_SHB = b"\x0a\x0d\x0d\x0a"
MAX_RECORD = 262144                  # largest capture length libpcap accepts for Ethernet
CHAIN = 16                           # consecutive plausible records that mark the end of a damaged region
CHUNK = 64 << 20
MAX_EVENTS = 1 << 16

# event kinds of the scan
GAP, RESUME, TAIL = 0, 1, 2


def sidecar(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".check.json")


@njit(cache=True)
def _u32(buf, i, big):  # pragma: no cover - compiled
    a, b, c, d = np.int64(buf[i]), np.int64(buf[i + 1]), np.int64(buf[i + 2]), np.int64(buf[i + 3])
    if big:
        return (a << 24) | (b << 16) | (c << 8) | d
    return (d << 24) | (c << 16) | (b << 8) | a


@njit(cache=True)
def _plausible(buf, p, big, t_lo, t_hi, frac_max, snap, min_incl):  # pragma: no cover - compiled
    ts = _u32(buf, p, big)
    fr = _u32(buf, p + 4, big)
    incl = _u32(buf, p + 8, big)
    orig = _u32(buf, p + 12, big)
    return t_lo <= ts <= t_hi and fr < frac_max and min_incl <= incl <= snap and incl <= orig


@njit(cache=True)
def _chain(buf, q, final, big, t_lo, t_hi, frac_max, snap, k):  # pragma: no cover - compiled
    """1 if k plausible records start at q (fewer suffice when they end exactly at the end of the file),
    0 if not, -1 if more data is needed to decide."""
    size = len(buf)
    p = q
    for i in range(k):
        if p + 16 > size:
            if final:
                return 1 if (p == size and i > 0) else 0
            return -1
        if not _plausible(buf, p, big, t_lo, t_hi, frac_max, snap, 1):
            return 0
        p += 16 + _u32(buf, p + 8, big)
        if p > size:
            return 0 if final else -1
    return 1


@njit(cache=True)
def _scan_window(buf, start, limit, final, state, big, t_lo, t_hi, frac_max, snap, k,
                 ev_pos, ev_kind):  # pragma: no cover - compiled
    """Walk records (state 0) or search the end of a damaged region (state 1) from ``start``.

    Positions at or beyond ``limit`` are left to the next window (whose data reaches further).
    Returns (stop position, state, complete records, number of events, finished).
    """
    size = len(buf)
    p, n_rec, n_ev = start, 0, 0
    while p < limit:
        if n_ev >= len(ev_pos) - 1:
            return p, state, n_rec, n_ev, False
        if state == 0:
            if p + 16 > size:                                  # final window: fewer bytes than a header
                ev_pos[n_ev] = p
                ev_kind[n_ev] = 2
                return p, state, n_rec, n_ev + 1, True
            if _plausible(buf, p, big, t_lo, t_hi, frac_max, snap, 0):
                nxt = p + 16 + _u32(buf, p + 8, big)
                if nxt <= size:
                    p = nxt
                    n_rec += 1
                    continue
                if final:                                      # the last record is incomplete
                    ev_pos[n_ev] = p
                    ev_kind[n_ev] = 2
                    return p, state, n_rec, n_ev + 1, True
                return p, state, n_rec, n_ev, False             # (cannot happen: the window reaches further)
            ev_pos[n_ev] = p                                   # the record chain breaks here
            ev_kind[n_ev] = 0
            n_ev += 1
            state = 1
            p += 1
        else:
            c = _chain(buf, p, final, big, t_lo, t_hi, frac_max, snap, k)
            if c == 1:
                ev_pos[n_ev] = p
                ev_kind[n_ev] = 1
                n_ev += 1
                state = 0
            elif c == -1:
                return p, state, n_rec, n_ev, False
            else:
                p += 1
    return p, state, n_rec, n_ev, final


@njit(cache=True)
def _walk_pcapng(buf, big):  # pragma: no cover - compiled
    pos, n, size = 0, 0, len(buf)
    while pos + 12 <= size:
        tl = _u32(buf, pos + 4, big)
        if tl < 12 or tl % 4 != 0:
            return pos, n, 1
        if pos + tl > size:
            break
        pos += tl
        n += 1
    return pos, n, 0


def _scan_pcap(fh, size: int, head: bytes, h) -> dict:
    big, ns = PCAP_MAGICS[head[:4]]
    snaplen = int.from_bytes(head[16:20], "big" if big else "little")
    snap = min(max(snaplen, 65535), MAX_RECORD)
    frac_max = 1_000_000_000 if ns else 1_000_000
    fh.seek(24)
    probe = fh.read(4 << 20)
    if len(probe) < 16:
        h.update(probe)
        return {"records": 0, "segments": [], "gaps": 0, "snaplen": snaplen, "valid_bytes": 24,
                "status": "complete" if size == 24 else "truncated last record"}
    # reference time: median time stamp of the first records (robust against a damaged first record)
    stamps, q = [], 0
    while q + 16 <= len(probe) and len(stamps) < 1001:
        stamps.append(int.from_bytes(probe[q:q + 4], "big" if big else "little"))
        incl = int.from_bytes(probe[q + 8:q + 12], "big" if big else "little")
        if incl > MAX_RECORD:
            break
        q += 16 + incl
    t0 = int(np.median(stamps))
    t_lo, t_hi = t0 - 86400, t0 + 3 * 86400
    look = CHAIN * (16 + snap)
    ev_pos = np.zeros(MAX_EVENTS, dtype=np.int64)
    ev_kind = np.zeros(MAX_EVENTS, dtype=np.int64)
    pos, state, records, hashed = 24, 0, 0, 24
    events: list[tuple[int, int]] = []
    while True:
        fh.seek(pos)
        data = fh.read(CHUNK + look)
        final = pos + len(data) >= size
        if hashed < pos + len(data):                           # every byte enters the digest once
            h.update(data[hashed - pos:])
            hashed = pos + len(data)
        buf = np.frombuffer(data, dtype=np.uint8)
        limit = len(data) if final else min(CHUNK, len(data))
        start = 0
        while True:
            p, state, n, n_ev, done = _scan_window(buf, start, limit, final, state, big, t_lo, t_hi, frac_max, snap,
                                                   CHAIN, ev_pos, ev_kind)
            records += int(n)
            events += [(pos + int(ev_pos[i]), int(ev_kind[i])) for i in range(n_ev)]
            if n_ev >= MAX_EVENTS - 1 and not done and p < limit:
                start = p                                       # event buffer full: continue in the same window
                continue
            break
        if done or final:
            end = pos + int(p)
            break
        pos += int(p)
    # valid segments from the events
    segs, a, ok = [], 24, True
    tail = None
    for off, kind in events:
        if kind == GAP and ok:
            segs.append([a, off])
            ok = False
        elif kind == RESUME and not ok:
            a, ok = off, True
        elif kind == TAIL:
            tail = off
    if ok:
        segs.append([a, tail if tail is not None else min(end, size)])
    segs = [s for s in segs if s[1] > s[0]]
    gaps = sum(1 for _, k in events if k == GAP)
    kept = sum(b - a for a, b in segs) + 24
    status = "complete"
    if gaps:
        status = "stray bytes between records"
    elif kept < size:
        status = "truncated last record"
    return {"records": records, "segments": segs, "gaps": gaps, "status": status, "snaplen": snaplen,
            "valid_bytes": kept}


def scan_capture(path: str | os.PathLike) -> dict:
    """Walk the records; report format, complete records, valid segments, outcome and the SHA-256 of the file."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        head = fh.read(24)
        if head[:4] in PCAP_MAGICS and len(head) == 24:
            h.update(head)
            out = {"format": "pcap", **_scan_pcap(fh, size, head, h)}
        elif head[:4] == PCAPNG_SHB:
            big = head[8:12] == b"\x1a\x2b\x3c\x4d"
            fh.seek(0)
            data_end, n, bad, pos = 0, 0, 0, 0
            carry = b""
            while True:
                chunk = fh.read(CHUNK)
                h.update(chunk)
                if not chunk:
                    break
                data = carry + chunk
                e, k, bad = _walk_pcapng(np.frombuffer(data, dtype=np.uint8), big) if len(data) >= 12 else (0, 0, 0)
                n += int(k)
                if bad:
                    for c in iter(lambda: fh.read(CHUNK), b""):
                        h.update(c)
                    data_end = pos + int(e)
                    break
                pos += int(e)
                carry = data[int(e):]
                data_end = pos
            out = {"format": "pcapng", "records": n, "segments": [[0, data_end]], "gaps": 0,
                   "status": "complete" if data_end == size else ("invalid block" if bad else "truncated last record"),
                   "valid_bytes": data_end}
        else:
            h.update(head)
            for c in iter(lambda: fh.read(CHUNK), b""):
                h.update(c)
            out = {"format": "unknown", "records": 0, "segments": [], "gaps": 0, "status": "not a capture",
                   "valid_bytes": size}
    out["size"] = size
    out["sha256"] = h.hexdigest()
    return out


def _rewrite(path: Path, segments: list[list[int]], header_bytes: int) -> None:
    tmp = path.with_name(path.name + ".repairing")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(src.read(header_bytes))
        for a, b in segments:
            src.seek(a)
            left = b - a
            while left > 0:
                c = src.read(min(CHUNK, left))
                if not c:
                    raise OSError(f"unexpected end of {path} while repairing")
                dst.write(c)
                left -= len(c)
    os.replace(tmp, path)


def check_and_repair(path: str | os.PathLike) -> dict:
    """Check a capture once and repair it in place if necessary (see the module documentation).

    Returns the stored outcome; ``repaired_now`` is True when this call changed the file.
    """
    path = Path(path)
    sc = sidecar(path)
    if sc.exists():
        info = json.loads(sc.read_text(encoding="utf-8"))
        if path.exists() and path.stat().st_size == info.get("final_size"):
            return {**info, "repaired_now": False}
    s = scan_capture(path)
    info = {"format": s["format"], "status": s["status"], "records": int(s["records"]),
            "stray_regions": int(s.get("gaps", 0)), "original_size": int(s["size"]),
            "original_sha256": s["sha256"], "repaired": False}
    if s["status"] in ("stray bytes between records", "truncated last record", "invalid block") and s["format"] != "unknown":
        if s["format"] == "pcap" and s.get("gaps"):
            _rewrite(path, s["segments"], 24)
        else:
            os.truncate(path, int(s["valid_bytes"]))
        info["repaired"] = True
    info["final_size"] = int(path.stat().st_size)
    info["removed_bytes"] = int(info["original_size"] - info["final_size"])
    sc.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return {**info, "repaired_now": info["repaired"]}


def stored_check(path: str | os.PathLike) -> dict | None:
    """The outcome of an earlier check of this capture, or None."""
    sc = sidecar(path)
    return json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else None

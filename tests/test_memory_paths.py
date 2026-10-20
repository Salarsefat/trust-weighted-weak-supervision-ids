"""The memory-saving paths of the corpus stage give the same result as the direct computation.

On the full corpus the stream has tens of millions of flows, so the corpus stage reads string columns straight
into categories, codes connection keys as integers and writes the stream in slices.  These tests compare each
path with the plain computation it replaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from twws.corpus import Workspace
from twws.extract import CATEGORY_COLUMNS, load_flows
from twws.snort import _canon_codes


def _flow_table(rng: np.random.Generator, n: int, ips: list[str], capture: str, day: str) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": [f"C{capture}{i}" for i in range(n)], "dir": rng.integers(0, 2, n).astype(np.int8),
        "ts": np.sort(rng.uniform(0, 1000, n)), "src_ip": rng.choice(ips, n), "dst_ip": rng.choice(ips, n),
        "orig_ip": rng.choice(ips, n), "resp_ip": rng.choice(ips, n), "src_port": rng.integers(1, 65535, n).astype(np.int32),
        "dst_port": rng.integers(1, 1024, n).astype(np.int32), "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "pkts": rng.integers(1, 50, n), "conn_state": rng.choice(["SF", "S0", "REJ"], n), "day": day,
        "capture": capture, "cap_id": np.int32(abs(hash(capture)) % 1000),
    })


def test_load_flows_equals_concatenation_then_categories(tmp_path):
    rng = np.random.default_rng(3)
    parts = []
    for k, (day, ips) in enumerate([("D1", ["10.0.0.9", "10.0.0.10", "172.31.1.2"]),
                                    ("D1", ["192.168.1.1", "10.0.0.10"]),
                                    ("D2", ["8.8.8.8", "10.0.0.2", "172.31.1.2", "10.0.0.10"])]):
        df = _flow_table(rng, 200 + 50 * k, ips, f"cap{k}", day)
        p = tmp_path / "flows" / day / f"cap{k}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        parts.append((p, df))
    ref = pd.concat([pd.read_parquet(p) for p, _ in sorted(parts, key=lambda x: str(x[0]))], ignore_index=True)
    for c in CATEGORY_COLUMNS:
        ref[c] = ref[c].astype("category")
    got = load_flows(tmp_path)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert list(got["src_ip"].cat.categories) == sorted(got["src_ip"].cat.categories)
    pd.testing.assert_frame_equal(load_flows(tmp_path, ["D2"]), ref[ref["day"] == "D2"].reset_index(drop=True)
                                  .assign(**{c: lambda d, c=c: d[c].cat.remove_unused_categories() for c in CATEGORY_COLUMNS}),
                                  check_exact=True)


def _string_keys(proto, a, ap, b, bp) -> np.ndarray:
    """The string key of the first version: protocol in lower case, endpoints ordered by address, then port."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    ap, bp = np.asarray(ap, dtype=np.int64), np.asarray(bp, dtype=np.int64)
    swap = (a > b) | ((a == b) & (ap > bp))
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    lop, hip = np.where(swap, bp, ap), np.where(swap, ap, bp)
    p = pd.Series(proto).astype(str).str.lower().to_numpy(dtype=object)
    return np.array([f"{x}|{y}|{z}|{u}|{v}" for x, y, z, u, v in zip(p, lo, lop, hi, hip)], dtype=object)


def test_integer_connection_keys_group_like_string_keys():
    rng = np.random.default_rng(5)
    ips = ["10.0.0.1", "10.0.0.2", "10.0.0.10", "9.9.9.9", "fe80::1"]
    tabs = []
    for n in (300, 500):
        tabs.append((pd.Series(rng.choice(["TCP", "tcp", "udp", "UDP"], n)), pd.Series(rng.choice(ips, n)),
                     pd.Series(rng.integers(1, 6, n)), pd.Series(rng.choice(ips, n)), pd.Series(rng.integers(1, 6, n))))
    codes = np.concatenate(_canon_codes(*tabs))
    strings = np.concatenate([_string_keys(*t) for t in tabs])
    # same partition of the rows: equal integer <=> equal string
    pairs = pd.DataFrame({"c": codes, "s": strings}).drop_duplicates()
    assert pairs["c"].is_unique and pairs["s"].is_unique
    # direction independence
    a = _canon_codes((["tcp"], ["10.0.0.2"], [80], ["10.0.0.1"], [5000]), (["TCP"], ["10.0.0.1"], [5000], ["10.0.0.2"], [80]))
    assert a[0][0] == a[1][0]


def test_stream_written_in_slices_reads_back_unchanged(tmp_path):
    rng = np.random.default_rng(7)
    stream = _flow_table(rng, 1000, ["10.0.0.1", "10.0.0.2", "172.31.0.5"], "capA", "D1")
    stream["day"] = rng.choice(["D1", "D2"], len(stream))
    stream["family"] = rng.choice(["Benign", "Bot"], len(stream))
    stream["attack"] = np.where(stream["family"] == "Bot", "Bot", "")
    stream["label"] = (stream["family"] == "Bot").astype(np.int8)
    for c in CATEGORY_COLUMNS:
        stream[c] = stream[c].astype("category")
    ws = Workspace(tmp_path / "w")
    ws.corpus_dir.mkdir(parents=True, exist_ok=True)
    ws.save_stream(stream, {"n": len(stream)}, chunk_rows=128)        # eight slices
    plain = pd.read_parquet(ws.stream_path)
    ref = stream.copy()
    for c in CATEGORY_COLUMNS:
        ref[c] = ref[c].astype(str)
    pd.testing.assert_frame_equal(plain, ref, check_exact=True)
    back = ws.load_stream()
    exp = plain.copy()
    for c in ["src_ip", "dst_ip", "orig_ip", "resp_ip", "proto", "conn_state", "day", "capture", "family", "attack"]:
        exp[c] = exp[c].astype("category")
    pd.testing.assert_frame_equal(back, exp, check_exact=True)

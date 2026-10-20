"""Diagnostics of the labelling rules (twws/experiments/diagnostics.py): the sub-stream of the attacker-controlled
endpoints and the inputs of the regularity rule under connection spacing."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import SCHEDULE
from twws.experiments.diagnostics import SUBSTREAM_COLUMNS, _regularity_inputs, extract_substream
from twws.experiments.extra import _transform_stream
from twws.oracles import MAL, BeaconOracle
from twws.pairs import pair_statistics

ATTACKER = "18.219.211.138"      # attacker of Thursday-15-02-2018 in the schedule of CSE-CIC-IDS2018
DAY = "Thursday-15-02-2018"


def _stream(n=21, iat=0.5, src=ATTACKER):
    """One TCP pair src -> 172.31.69.25:22 with a connection every ``iat`` seconds, both directions recorded, and
    one unrelated benign pair."""
    rows = []
    for k in range(n):
        for d, (a, b, sp, dp) in enumerate(((src, "172.31.69.25", 40000 + k, 22), ("172.31.69.25", src, 22, 40000 + k))):
            rows.append({"conn": k, "dir": d, "ts": 100.0 + iat * k, "t_first": 100.0 + iat * k,
                         "t_last": 100.0 + iat * k + 0.1, "src_ip": a, "dst_ip": b, "src_port": sp, "dst_port": dp,
                         "proto": "tcp", "day": DAY, "label": 1, "family": "BruteForce", "attack": "SSH-Bruteforce",
                         "pkts": 10, "bytes": 1000, "conn_duration": 0.1})
    for k in range(5):
        rows.append({"conn": 1000 + k, "dir": 0, "ts": 50.0 + 7.0 * k, "t_first": 50.0 + 7.0 * k,
                     "t_last": 50.5 + 7.0 * k, "src_ip": "172.31.64.10", "dst_ip": "172.31.0.2", "src_port": 50000 + k,
                     "dst_port": 53, "proto": "udp", "day": DAY, "label": 0, "family": "Benign", "attack": "Benign",
                     "pkts": 2, "bytes": 120, "conn_duration": 0.01})
    return pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)


def test_extract_substream_keeps_records_of_controlled_endpoints_in_order(tmp_path):
    st = _stream(n=4)
    st["extra_column"] = 1.0
    src = tmp_path / "stream.parquet"
    pq.write_table(pa.Table.from_pandas(st, preserve_index=False), src, row_group_size=3)   # several row groups
    out = tmp_path / "sub.parquet"
    res = extract_substream(SimpleNamespace(corpus=SimpleNamespace(schedule=SCHEDULE)), src, out)
    sub = pd.read_parquet(out)
    keep = st["src_ip"].eq(ATTACKER) | st["dst_ip"].eq(ATTACKER)
    assert list(sub.columns) == SUBSTREAM_COLUMNS
    assert res["records"] == len(sub) == int(keep.sum()) == 8
    assert sub[["conn", "dir"]].values.tolist() == st.loc[keep, ["conn", "dir"]].values.tolist()
    assert ATTACKER in res["endpoints"] and res["stream_records"] == len(st)


def test_spacing_lifts_short_intervals_above_the_minimum_of_the_regularity_rule():
    st = _stream(n=21, iat=0.5)
    ctrl = (st["label"].eq(1) & st["src_ip"].eq(ATTACKER) & st["day"].eq(DAY)).to_numpy()
    beacon = BeaconOracle(min_conn=10, disp_max=0.1, iat_min=1.0)
    shares = {}
    for level in (1.0, 2.0):
        s = st if level == 1.0 else _transform_stream(st, ctrl, "throttle", level)
        ps = pair_statistics(s)
        aux = ps.iloc[pd.Index(ps["conn"].to_numpy()).get_indexer(s["conn"].to_numpy())].reset_index(drop=True)
        ri = _regularity_inputs(aux, ctrl, beacon)
        votes = beacon.vote(aux)[ctrl]
        shares[level] = (ri, float(np.mean(votes == MAL)))
    ri1, mal1 = shares[1.0]
    ri2, mal2 = shares[2.0]
    # spacing multiplies the intervals: the relative dispersion stays 0, the median interval moves from 0.5 s to 1 s
    assert ri1["iat_med_below_min"] == 1.0 and ri2["iat_med_below_min"] == 0.0
    assert ri1["disp_at_most_max"] == ri2["disp_at_most_max"] == 1.0
    assert ri1["iat_med_quantiles"]["0.5"] == 0.5 and ri2["iat_med_quantiles"]["0.5"] == 1.0
    # the rule needs ten connections in the last hour: from the tenth connection of the pair on it votes malicious
    assert mal1 == 0.0 and mal2 > 0.5

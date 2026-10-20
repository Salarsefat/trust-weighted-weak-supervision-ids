"""Pair statistics of the oracles ``rate`` and ``beacon`` (twws.pairs, docs/ORACLES_EXTENDED.md)."""
import numpy as np
import pandas as pd

from twws.oracles import ABS, BEN, MAL, BeaconOracle, RateOracle
from twws.pairs import PAIR_COLUMNS, pair_features


def _rec(conn, d, ts, src, dst, sport, dport, proto="tcp"):
    return {"conn": conn, "dir": d, "ts": ts, "src_ip": src, "dst_ip": dst, "src_port": sport,
            "dst_port": dport, "proto": proto}


def _stream():
    rows = []
    c = 0
    # pair A: 10.0.0.1 -> 10.0.0.9:80, a connection every 10 s from t=0 to t=200 (21 connections),
    # both directions recorded
    for k in range(21):
        rows.append(_rec(c, 0, 10.0 * k, "10.0.0.1", "10.0.0.9", 40000 + k, 80))
        rows.append(_rec(c, 1, 10.0 * k, "10.0.0.9", "10.0.0.1", 80, 40000 + k))
        c += 1
    # pair B: two simultaneous connections at t=5 to port 22, only the reply record of the second one
    rows.append(_rec(c, 0, 5.0, "10.0.0.2", "10.0.0.9", 50000, 22))
    c += 1
    rows.append(_rec(c, 1, 5.0, "10.0.0.9", "10.0.0.2", 22, 50001))
    c += 1
    # same hosts as pair A, other port: a separate pair
    rows.append(_rec(c, 0, 100.0, "10.0.0.1", "10.0.0.9", 40100, 443))
    c += 1
    # UDP: no pair
    rows.append(_rec(c, 0, 7.0, "10.0.0.1", "10.0.0.53", 5353, 53, proto="udp"))
    return pd.DataFrame(rows)


def test_pair_counts_windows_and_regularity():
    st = _stream()
    pf = pair_features(st)
    assert list(pf.columns) == PAIR_COLUMNS
    a = st.index[(st["dst_port"] == 80) | (st["src_port"] == 80)]
    last = st.index[(st["ts"] == 200.0) & (st["dir"] == 0)][0]
    assert pf.loc[last, "aux_pair_n60"] == 7          # starts at 140, 150, ..., 200
    assert pf.loc[last, "aux_pair_n1h"] == 21
    assert pf.loc[last, "aux_pair_iat_med"] == 10.0
    assert pf.loc[last, "aux_pair_iat_disp"] == 0.0
    # both directions of a connection carry the same values and the same pair id
    rep = st.index[(st["ts"] == 200.0) & (st["dir"] == 1)][0]
    assert pf.loc[rep].equals(pf.loc[last])
    assert pf.loc[a, "aux_pair_id"].nunique() == 1
    # the first connection of a pair has no inter-arrival time
    first = st.index[(st["ts"] == 0.0) & (st["dir"] == 0)][0]
    assert pf.loc[first, "aux_pair_n60"] == 1 and np.isnan(pf.loc[first, "aux_pair_iat_med"])


def test_simultaneous_starts_reply_only_records_ports_and_udp():
    st = _stream()
    pf = pair_features(st)
    b = st.index[(st["dst_port"] == 22) | (st["src_port"] == 22)]
    # a reply-only record is mapped to the pair (originator, responder, responder port)
    assert pf.loc[b, "aux_pair_id"].nunique() == 1
    assert (pf.loc[b, "aux_pair_n60"] == 2).all()      # simultaneous starts count for both
    p443 = st.index[st["dst_port"] == 443][0]
    p80 = st.index[st["dst_port"] == 80][0]
    assert pf.loc[p443, "aux_pair_id"] != pf.loc[p80, "aux_pair_id"]
    assert pf.loc[p443, "aux_pair_n60"] == 1
    udp = st.index[st["proto"] == "udp"][0]
    assert pf.loc[udp, "aux_pair_id"] == -1 and np.isnan(pf.loc[udp, "aux_pair_n60"])


def test_rows_subset_keeps_order():
    st = _stream()
    full = pair_features(st)
    rows = np.array([5, 0, 44, 3])
    sub = pair_features(st, rows)
    assert sub.reset_index(drop=True).equals(full.iloc[rows].reset_index(drop=True))


def test_rate_oracle_threshold_over_pairs_and_votes():
    # 1000 benign pairs with peak 1..10, one flood pair whose many rows must not move the threshold
    pid = np.concatenate([np.arange(1000), np.full(5000, 1000)])
    n60 = np.concatenate([1 + np.arange(1000) % 10, np.full(5000, 900.0)])
    ref = np.ones(len(pid), dtype=bool)
    thr = RateOracle.pair_threshold(pid, n60.astype(float), ref, 0.99)
    assert thr == np.quantile(np.r_[1 + np.arange(1000) % 10, 900.0], 0.99)
    assert thr <= 10.0
    aux = pd.DataFrame({"aux_pair_n60": [1.0, 10.0, 10.4, 11.0, 900.0, np.nan]})
    v = RateOracle(threshold=10.0, guard=0.05).vote(aux)
    assert v.tolist() == [BEN, ABS, ABS, MAL, MAL, ABS]


def test_beacon_oracle_votes():
    aux = pd.DataFrame({
        "aux_pair_n1h": [12, 12, 12, 12, 5, 12],
        "aux_pair_iat_med": [30.0, 30.0, 30.0, 0.5, 30.0, np.nan],
        "aux_pair_iat_disp": [0.05, 0.15, 0.5, 0.0, 0.0, np.nan],
    })
    v = BeaconOracle(min_conn=10, disp_max=0.1, iat_min=1.0).vote(aux)
    # regular -> malicious; in the band -> abstain; irregular -> benign; too fast or too few -> abstain
    assert v.tolist() == [MAL, ABS, BEN, ABS, ABS, ABS]


def test_grouped_label_model_merges_dependent_oracles_and_uses_independent_evidence():
    from twws.weak import aggregate_votes, dependent_groups, merge_group_votes

    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.random(n) < 0.2).astype(int)
    kind = rng.integers(0, 2, n)                      # two kinds of attack
    # oracles A and B: copies of each other, fire on attack kind 0 only (and on 1 % of benign flows)
    a = np.where((y == 1) & (kind == 0), 1, np.where(rng.random(n) < 0.01, 1, 0))
    b = a.copy()
    # oracle C: fires on both kinds (90 %) and on 0.5 % of benign flows
    c = np.where(y == 1, (rng.random(n) < 0.9).astype(int), (rng.random(n) < 0.005).astype(int))
    V = np.column_stack([a, b, c]).astype(np.int8)
    grp = dependent_groups(V, None, 0.5)
    assert grp["groups"] == [[0, 1], [2]]
    Vg = merge_group_votes(V, grp["groups"])
    assert Vg.shape == (n, 2) and (Vg[:, 0] == a).all()
    g = aggregate_votes(V, "ds_grouped", names=["A", "B", "C"])
    assert g["model"]["sources"] == ["A+B", "C"]
    rec = ((g["label"] == 1) & (y == 1)).sum() / (y == 1).sum()
    assert rec > 0.85                                  # attack kind 1 is recovered through oracle C
    assert g["model"]["start"] in [r["start"] for r in g["model"]["starts"]]

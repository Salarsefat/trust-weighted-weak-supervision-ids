"""Schedule-based labels and partition protocols."""
import numpy as np
import pandas as pd

from twws.corpus import TEST, TRAIN, UNUSED, VAL_A, VAL_B, assemble_stream, make_split, sample_corpus
from twws.labels import estimate_utc_offset, label_flows

from conftest import SCHEDULE


def test_utc_offset_is_recovered_from_traffic(synth_flows, schedule):
    off, diag = estimate_utc_offset(synth_flows, schedule)
    assert off == -4
    shifted = synth_flows.copy()
    shifted["ts"] = shifted["ts"] + 2 * 3600          # same local schedule, clock two hours ahead
    off2, _ = estimate_utc_offset(shifted, schedule)
    assert off2 == -6                                  # local time = UTC + offset


def test_labels_follow_attacker_addresses(synth_flows, schedule):
    lab = label_flows(synth_flows, schedule, -4)
    assert set(lab["label"].unique()) <= {-1, 0, 1}
    fam = lab["family"].value_counts()
    for f in ("BruteForce", "DoS", "DDoS", "Web", "Infiltration", "Bot"):
        assert fam.get(f, 0) > 0
    assert "Heartbleed" not in fam.index                      # no Heartbleed entry in the official schedule
    assert (lab.loc[lab["family"] == "Ambiguous", "label"] == -1).all()
    mal = lab["label"].to_numpy() == 1
    assert lab.loc[mal, "in_window"].mean() > 0.99


def _corpus(synth_flows):
    stream, info = assemble_stream(synth_flows.copy(), SCHEDULE)
    keep, rep = sample_corpus(stream, family_cap=600, benign_share=0.83, max_rows=None, seed=42)
    meta = stream.loc[keep].reset_index(drop=True)
    return stream, meta, rep


def test_sampling_rule(synth_flows):
    stream, meta, rep = _corpus(synth_flows)
    assert (meta["family"] != "Ambiguous").all()
    for f, n in rep["selected"].items():
        if f != "Benign":
            assert n == min(600, rep["available"][f])
    assert abs(rep["malicious_share"] - 0.17) < 0.005
    keep2, rep2 = sample_corpus(stream, family_cap=600, benign_share=0.83, max_rows=None, seed=42)
    assert rep2["selected"] == rep["selected"]                 # deterministic for a given seed


def test_partition_protocols(synth_flows):
    _, meta, _ = _corpus(synth_flows)
    # random: class-proportional 60/20/20
    s = make_split(meta, "random", 42)
    for f in meta["family"].unique():
        m = (meta["family"] == f).to_numpy()
        assert abs((s[m] == TRAIN).mean() - 0.6) < 0.02
        assert abs((s[m] == TEST).mean() - 0.2) < 0.02
    # conn: the two directions of a connection never straddle subsets
    s = make_split(meta, "conn", 42)
    assert pd.Series(s).groupby(meta["conn"].to_numpy()).nunique().max() == 1
    # chrono: within class and day, every training connection starts before every test connection
    s = make_split(meta, "chrono", 42)
    for (f, d), g in meta.assign(split=s).groupby(["family", "day"], observed=True):
        tr, te = g.loc[g["split"] == TRAIN, "ts"], g.loc[g["split"] == TEST, "ts"]
        if len(tr) and len(te):
            first = g.groupby("conn")["ts"].min()
            assert first[g.loc[g["split"] == TRAIN, "conn"].unique()].max() <= first[g.loc[g["split"] == TEST, "conn"].unique()].min()
    # forward: whole days
    days = sorted(meta["day"].astype(str).unique())
    fw = {"train": days[:4], "val": days[4:6], "test": days[6:]}
    s = make_split(meta, "forward", 42, fw)
    d = meta["day"].astype(str).to_numpy()
    assert set(d[s == TRAIN]) <= set(fw["train"]) and set(d[s == TEST]) <= set(fw["test"])
    assert set(d[(s == VAL_A) | (s == VAL_B)]) <= set(fw["val"])
    # leave-one-attack-out: the family is absent from training and validation
    s = make_split(meta, "loao:Bot", 42)
    bot = (meta["family"] == "Bot").to_numpy()
    assert set(np.unique(s[bot])) <= {TEST, UNUSED} and (s[bot] == TEST).any()


def test_deduplication_keeps_reused_five_tuples():
    """Two hosts, both captured: every connection is seen twice.  A second connection that reuses the
    five-tuple within the tolerance must survive once (no chain removal), in either arrival order."""
    import pandas as pd

    from twws.extract import deduplicate

    def conn(uid, cap, ts):
        base = {"uid": uid, "cap_id": cap, "ts": ts, "proto": "tcp", "orig_ip": "10.0.0.1", "resp_ip": "10.0.0.2"}
        return [{**base, "dir": 0, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 40000, "dst_port": 80},
                {**base, "dir": 1, "src_ip": "10.0.0.2", "dst_ip": "10.0.0.1", "src_port": 80, "dst_port": 40000}]

    A, B = 11, 22
    rows = (conn("x", A, 100.000) + conn("x", B, 100.002)          # connection X, seen in both captures
            + conn("y", B, 100.500) + conn("y", A, 100.503)        # Y reuses the five-tuple 0.5 s later
            + conn("z", A, 100.900)                                # Z seen in capture A only
            + conn("w", A, 500.000) + conn("w", B, 499.999))       # W, later, arrival order reversed
    flows = pd.DataFrame(rows)
    out, stats = deduplicate(flows, tol_s=1.0)
    assert stats["connections_before"] == 7 and stats["connections_removed"] == 3
    kept = out.groupby(["uid", "cap_id"]).size()
    assert len(out) == 8 and (kept == 2).all()                      # both directions from the same capture
    assert sorted(out["uid"].unique()) == ["w", "x", "y", "z"]
    assert set(kept.index) == {("x", A), ("y", B), ("z", A), ("w", B)}
    # without capture information nothing can be identified as a second observation
    out2, stats2 = deduplicate(flows.assign(cap_id=0), tol_s=1.0)
    assert stats2["connections_removed"] == 0 and len(out2) == len(flows)


def test_flow_is_named_after_the_attack_whose_victim_it_reaches(tmp_path):
    """A day with two attacks of one attacker (the shape of the CICIDS2017 Friday): bots that contact the attacker
    outside their window keep the botnet family; the nearest window decides only between attacks that do not list
    the flow's endpoint as a victim (or that both do)."""
    from twws.labels import load_schedule

    p = tmp_path / "s.yaml"
    p.write_text(
        "dataset: X\ndefault_utc_offset_hours: 0\nwindow_slack_seconds: 60\nnat_address: '172.16.0.1'\n"
        "attacker_addresses: ['9.9.9.9']\n"
        "days:\n  Friday:\n    date: '2017-07-07'\n    family: Bot\n    attacks:\n"
        "      - {name: Bot, family: Bot, start: '10:00', end: '11:00', attackers: ['9.9.9.9'], victims: ['10.0.0.15', '10.0.0.9']}\n"
        "      - {name: PortScan, family: PortScan, start: '14:00', end: '15:00', attackers: ['9.9.9.9'], victims: ['10.0.0.50']}\n")
    sched = load_schedule(p)
    t0 = sched["days"]["Friday"]["midnight_utc"]
    rows = [  # (src, dst, local time in hours)
        ("10.0.0.15", "9.9.9.9", 14.5),   # bot beacon inside the scan window -> Bot (outside its own window)
        ("10.0.0.9", "9.9.9.9", 10.5),    # bot beacon inside its window -> Bot
        ("9.9.9.9", "10.0.0.50", 14.5),   # scan of the scan victim -> PortScan
        ("172.16.0.1", "10.0.0.50", 14.5),  # translated scan traffic inside the window -> PortScan
        ("172.16.0.1", "10.0.0.50", 12.0),  # translated traffic outside every window -> benign
        ("9.9.9.9", "10.0.0.77", 12.9),   # attacker flow that reaches no listed victim -> nearest window (PortScan)
        ("10.0.0.3", "8.8.8.8", 14.5),    # unrelated -> benign
    ]
    f = pd.DataFrame({"day": "Friday", "ts": [t0 + h * 3600 for *_, h in rows],
                      "src_ip": [r[0] for r in rows], "dst_ip": [r[1] for r in rows],
                      "orig_ip": [r[0] for r in rows], "resp_ip": [r[1] for r in rows]})
    lab = label_flows(f, sched, 0)
    assert lab["family"].tolist() == ["Bot", "Bot", "PortScan", "PortScan", "Benign", "PortScan", "Benign"]
    assert lab["in_window"].tolist() == [False, True, True, True, False, False, False]
    assert lab["label"].tolist() == [1, 1, 1, 1, 0, 1, 0]

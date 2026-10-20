"""Trace-driven emulation of the quality of service under enforcement (twws/qos.py)."""
import numpy as np
import pytest

from twws.qos import dropped_share, emulate, latency_for_rate, merged_blocks, spread_over_bins


def test_blocks_of_one_source_are_merged_and_sources_kept_apart():
    ms, ma, mb, cum = merged_blocks(np.array([1, 1, 1, 2]), np.array([10.0, 12.0, 30.0, 11.0]), hold=5.0)
    assert ms.tolist() == [1, 1, 2]
    assert ma.tolist() == [10.0, 30.0, 11.0] and mb.tolist() == [17.0, 35.0, 16.0]
    assert cum.tolist() == [0.0, 7.0, 0.0]          # blocked time of the same source before each interval


def test_dropped_share_is_the_overlap_with_the_blocks_of_the_own_source():
    blocks = merged_blocks(np.array([1, 2]), np.array([10.0, 0.0]), hold=5.0)        # source 1: [10, 15)
    src = np.array([1, 1, 1, 1, 2, 3])
    t0 = np.array([0.0, 12.0, 14.0, 20.0, 1.0, 11.0])
    t1 = np.array([5.0, 12.0, 18.0, 21.0, 1.0, 12.0])
    f = dropped_share(blocks, src, t0, t1)
    assert f.tolist() == pytest.approx([0.0, 1.0, 0.25, 0.0, 1.0, 0.0])


def test_spread_over_bins_conserves_the_amount():
    t0 = np.array([0.2, 0.5, 3.0])
    t1 = np.array([0.7, 3.5, 3.0])
    out = spread_over_bins(t0, t1, np.array([10.0, 30.0, 5.0]), n_bins=5)
    assert out.sum() == pytest.approx(45.0)
    assert out[0] == pytest.approx(10.0 + 5.0)       # 0.5 s of the 3-s record falls into bin 0
    assert out[1] == pytest.approx(10.0) and out[2] == pytest.approx(10.0) and out[3] == pytest.approx(5.0 + 5.0)


def _day(pred_attack=True, pred_benign_fp=False):
    # source 1 attacks (records every second), source 2 is benign; one benign record of source 1 after the attack
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.5, 2.5, 6.0])
    return {"src": np.array([1, 1, 1, 1, 1, 2, 2, 1]), "ts": ts, "t_avail": ts + 0.1,
            "duration": np.zeros(8), "bytes": np.full(8, 100.0), "pkts": np.full(8, 2.0),
            "label": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "pred": np.array([pred_attack] * 5 + [pred_benign_fp, False, False])}


def test_emulation_blocks_attack_traffic_and_reports_the_collateral_damage():
    r = emulate([_day()], speed=1.0, latency_s=0.0, hold_s=5.0, attack=True)
    # the first attack record is delivered (its verdict comes after it); the next four fall into the block
    assert r["attack_bytes_dropped_share"] == pytest.approx(0.8)
    # the benign record of source 1 at t = 6 s falls into the block renewed by the verdict of t = 4 s
    assert r["throughput_preservation"] == pytest.approx(2 / 3)
    assert r["availability"] == pytest.approx(2 / 3)
    r0 = emulate([_day()], speed=1.0, latency_s=0.0, hold_s=5.0, attack=False)
    assert r0["throughput_preservation"] == 1.0 and r0["packet_loss_ratio"] == 0.0 and r0["attack_bytes_dropped_share"] is None


def test_compression_does_not_compress_the_hold_down():
    slow = emulate([_day()], speed=1.0, latency_s=0.0, hold_s=1.5, attack=True)
    fast = emulate([_day()], speed=4.0, latency_s=0.0, hold_s=1.5, attack=True)
    assert fast["attack_bytes_dropped_share"] >= slow["attack_bytes_dropped_share"]
    assert fast["offered_benign_mbps"] > slow["offered_benign_mbps"]


def test_latency_is_taken_from_the_smallest_sustained_rate_at_or_above_the_load():
    rows = [{"workers": 1, "offered_rate": 1000, "saturated": False, "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0}},
            {"workers": 2, "offered_rate": 1000, "saturated": False, "latency_ms": {"p50": 1.0, "p95": 1.5, "p99": 3.0}},
            {"workers": 1, "offered_rate": 5000, "saturated": True, "latency_ms": {"p50": 9.0, "p95": 99.0, "p99": 120.0}}]
    assert latency_for_rate(rows, 10.0)["p95_ms"] == 1.5
    assert latency_for_rate(rows, 4000.0)["within_capacity"] is False


@pytest.mark.slow
def test_qos_experiment_on_a_synthetic_workspace(tmp_path):
    """The experiment classifies every stream record and emulates both conditions at every speed."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    common = ["--config", str(repo / "configs" / "smoke.yaml"), "--set", f"work_dir={work}",
              "--set", f"transfer.work_dir={tmp_path / 'src'}", "--set", "synthetic.n_benign_per_day=2000",
              "--set", "corpus.family_cap=800", "--set", "experiments.seeds=[7]", "--set", "experiments.pipeline_seeds=[]",
              "--set", "search.n_trials=2", "--set", "search.n_random=2", "--set", "experiments.qos.speeds=[1,4]",
              "--set", "experiments.qos.holds=[1,5]"]

    def cli(*args):
        r = subprocess.run([sys.executable, "-m", "twws.cli", *args, *common], cwd=repo, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]

    cli("synthetic")
    cli("run", "search", "qos", "--n-jobs", "2")
    q = json.loads((work / "results" / "qos.json").read_text())
    assert q["verdicts"]["records"] > q["verdicts"]["records_with_ground_truth"] * 0.5
    assert len(q["rows"]) == 4 and len(q["hold_rows"]) == 4
    for r in q["rows"]:
        assert 0.0 <= r["throughput_preservation"] <= 1.0 and 0.0 <= r["availability"] <= 1.0
        assert r["offered_benign_mbps"] > 0
    att = [r for r in q["rows"] if r["attack"]]
    assert all(r["attack_bytes_dropped_share"] is not None for r in att)

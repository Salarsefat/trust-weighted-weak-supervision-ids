"""Zeek extraction and the Snort wrapper on a small capture with known content.

Skipped when Zeek, Snort or scapy are not installed.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import have

scapy = pytest.importorskip("scapy.all")

A, B, C = "172.31.64.10", "172.31.69.25", "172.31.64.11"
TOKEN = b"twws-test-token"


def _tcp_session(t0, src, dst, sport, dport, c2s: list[bytes], s2c: list[bytes]):
    """A complete TCP connection: handshake, alternating data segments, orderly close."""
    from scapy.all import IP, TCP, Raw

    pk, t = [], t0
    cs, ss = 1000, 5000

    def add(p, dt=0.01):
        nonlocal t
        t += dt
        p.time = t
        pk.append(p)

    add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="S", seq=cs))
    add(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="SA", seq=ss, ack=cs + 1))
    cs += 1
    ss += 1
    add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", seq=cs, ack=ss))
    for i in range(max(len(c2s), len(s2c))):
        if i < len(c2s):
            add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="PA", seq=cs, ack=ss) / Raw(c2s[i]))
            cs += len(c2s[i])
            add(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="A", seq=ss, ack=cs))
        if i < len(s2c):
            add(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="PA", seq=ss, ack=cs) / Raw(s2c[i]))
            ss += len(s2c[i])
            add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", seq=cs, ack=ss))
    add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="FA", seq=cs, ack=ss))
    add(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="FA", seq=ss, ack=cs + 1))
    add(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", seq=cs + 1, ack=ss + 1))
    return pk


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

    rng = np.random.default_rng(0)
    t0 = 1518610000.0
    rnd = rng.integers(0, 256, 1200, dtype=np.uint8).tobytes()
    pk = []
    # connection 1: constant payload one way (entropy 0), random payload back (entropy near 8)
    pk += _tcp_session(t0, A, B, 40001, 80, [b"a" * 500, b"a" * 300], [rnd, rnd[:800]])
    # connection 2: carries the token the test rule looks for
    pk += _tcp_session(t0 + 5, C, B, 40002, 8080, [b"GET /" + TOKEN + b" HTTP/1.1\r\n\r\n"], [b"HTTP/1.1 200 OK\r\n\r\n"])
    # three SYNs without any answer (state S0)
    for i in range(3):
        p = IP(src=A, dst=B) / TCP(sport=40010, dport=81, flags="S", seq=77)
        p.time = t0 + 10 + i
        pk.append(p)
    # a UDP exchange
    p = IP(src=A, dst=B) / UDP(sport=50000, dport=53) / Raw(b"q" * 40)
    p.time = t0 + 20
    pk.append(p)
    p = IP(src=B, dst=A) / UDP(sport=53, dport=50000) / Raw(b"r" * 90)
    p.time = t0 + 20.01
    pk.append(p)
    pk.sort(key=lambda x: float(x.time))
    path = tmp_path_factory.mktemp("cap") / "test.pcap"
    frames = []
    for x in pk:
        f = Ether() / x
        f.time = x.time                     # the frame is a new object: carry the intended timestamp over
        frames.append(f)
    wrpcap(str(path), frames)
    return path


@pytest.mark.skipif(not have("zeek"), reason="zeek not installed")
def test_zeek_extraction_counts(capture, tmp_path):
    from twws.extract import flows_from_log, run_zeek, zeek_binary

    logp = run_zeek(capture, tmp_path / "zeek", zeek_binary())
    fl = flows_from_log(logp)
    for c in ("src_ip", "dst_ip", "proto"):
        fl[c] = fl[c].astype(str)
    key = lambda s, sp: fl[(fl.src_ip == s) & (fl.src_port == sp)].iloc[0]
    c2s, s2c = key(A, 40001), key(B, 80)
    # originator: SYN, ACK, 2 data, 2 ACKs of server data, FIN, last ACK = 8 packets
    assert (c2s.pkts, c2s.syn, c2s.fin, c2s.psh, c2s.pay) == (8, 1, 1, 2, 800)
    assert (s2c.pkts, s2c.syn, s2c.fin, s2c.psh, s2c.pay) == (6, 1, 1, 2, 2000)
    assert c2s.rev_pkts == s2c.pkts and s2c.rev_bytes == c2s.bytes
    assert c2s.dir == 0 and s2c.dir == 1 and c2s.uid == s2c.uid
    assert abs(c2s.ts - (1518610000.0 + 0.01)) < 1e-3            # connection start = time of the SYN
    assert abs(c2s.duration - 0.13) < 1e-3 and abs(s2c.duration - 0.11) < 1e-3   # first to last packet per direction
    assert abs(c2s.entropy) < 1e-6 and s2c.entropy > 7.5
    assert c2s.cap == c2s.pay                                    # payload fully captured
    syn_only = key(A, 40010)
    assert (syn_only.pkts, syn_only.syn, syn_only.rev_pkts) == (3, 3, 0)
    assert len(fl[(fl.src_ip == B) & (fl.src_port == 81)]) == 0  # the silent direction yields no flow
    assert np.isnan(syn_only.entropy)
    udp = fl[fl.proto == "udp"].sort_values("dir")
    assert udp.pkts.tolist() == [1, 1] and udp.pay.tolist() == [40, 90]
    assert len(fl) == 2 + 2 + 1 + 2


@pytest.mark.skipif(not (have("zeek") and have("snort")), reason="zeek or snort not installed")
def test_snort_alerts_are_mapped_to_connections(capture, tmp_path):
    from twws.corpus import connection_table
    from twws.extract import connection_ids, flows_from_log, run_zeek, zeek_binary
    from twws.snort import alerts_to_connections, build_conf, parse_alerts, run_snort, snort_binary

    rule = tmp_path / "test.rules"
    rule.write_text(f'alert tcp any any -> any 8080 (msg:"TWWS test token"; content:"{TOKEN.decode()}"; sid:1000001; rev:1;)\n')
    info = build_conf([str(rule)], tmp_path / "snort.conf")
    assert info["active_rules"] == 1 and info["rule_source"] == "configured"
    alerts = parse_alerts(run_snort(capture, tmp_path / "snort", info["conf"], snort_binary()))
    assert len(alerts) >= 1 and (alerts["sid"] == 1000001).all()
    fl = flows_from_log(run_zeek(capture, tmp_path / "zeek", zeek_binary()))
    fl["cap_id"] = 0
    fl["conn"] = connection_ids(fl)
    conns = connection_table(fl)
    mapped = alerts_to_connections(alerts, conns, slack_s=1.0)
    hit = set(mapped.loc[mapped["snort_alert"], "conn"])
    assert len(hit) == 1
    row = conns[conns["conn"].isin(hit)].iloc[0]
    assert (str(row.orig_ip), int(row.resp_port)) == (C, 8080)


@pytest.mark.skipif(not (have("zeek") and have("snort")), reason="zeek or snort not installed")
def test_extraction_stages_with_the_configuration_of_the_paper(capture, tmp_path):
    """Zeek and Snort stages exactly as configs/paper.yaml sets them: worker counts left to the machine
    (``n_jobs: null``) and the rule files enabled in the distribution's snort.conf (``snort.rules: []``)."""
    from conftest import REPO
    from twws.config import load_config
    from twws.extract import extract_manifest
    from twws.snort import snort_manifest
    from twws.utils import worker_count

    cfg = load_config(str(REPO / "configs" / "paper.yaml"))
    assert cfg.get_path("extract.n_jobs") is None and cfg.get_path("snort.n_jobs") is None
    assert worker_count(None) >= 1 and worker_count(3) == 3 and worker_count(-1) == worker_count(None)
    man = pd.DataFrame([{"day": "Wednesday-14-02-2018", "member": "pcap/test.pcap", "local_path": str(capture)}])
    rep = extract_manifest(cfg, man, tmp_path, tmp_path / "extract")
    assert len(rep) == 1 and int(rep["flows"].iloc[0]) == 7
    info = snort_manifest(cfg, man, tmp_path, tmp_path / "snort")
    assert info["rule_source"] == "distribution default"
    assert info["active_rules"] > 0, "the distribution's snort.conf enables no rule file"
    assert len(info["captures"]) == 1
    assert (tmp_path / "snort" / "alerts" / "Wednesday-14-02-2018" / "test.pcap.parquet").exists()


@pytest.mark.skipif(not have("zeek"), reason="zeek not installed")
def test_tls_and_self_signed_certificates(tmp_path):
    """Three real handshakes recorded on a loopback interface with throw-away certificates:
    port 8443 self-signed over TLS 1.2, port 9443 CA-signed over TLS 1.2, port 7443 self-signed
    over TLS 1.3 (certificate encrypted, so its issuer cannot be observed)."""
    from pathlib import Path

    from twws.extract import flows_from_log, run_zeek, zeek_binary

    pcap = Path(__file__).parent / "data" / "tls_handshakes.pcap"
    fl = flows_from_log(run_zeek(pcap, tmp_path / "zeek", zeek_binary()))
    by_port = {int(r.dst_port): r for r in fl[fl["dir"] == 0].itertuples()}
    assert all(by_port[p].tls == 1 for p in (8443, 9443, 7443))
    assert by_port[8443].tls_self_signed == 1.0
    assert by_port[9443].tls_self_signed == 0.0
    assert np.isnan(by_port[7443].tls_self_signed)

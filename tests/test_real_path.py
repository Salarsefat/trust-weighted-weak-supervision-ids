"""The path that real data takes: archives -> ranged retrieval -> Zeek + Snort -> stream -> corpus -> oracles.

Two miniature "capture days" are generated with scapy (per-host captures, attack traffic from the
scheduled attacker addresses inside the scheduled windows, benign traffic, one connection seen
from two capture points) and served by a local range-capable HTTP server.
"""
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, SCHEDULE, have

scapy = pytest.importorskip("scapy.all")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (have("zeek") and have("snort")), reason="zeek or snort not installed")]

from test_tools import TOKEN, _tcp_session  # noqa: E402

UTC_OFFSET = -4
DAY_BF, DAY_BOT = "Wednesday-14-02-2018", "Friday-02-03-2018"
MIDNIGHT = {DAY_BF: 1518566400.0, DAY_BOT: 1519948800.0}             # 00:00 UTC of the capture day
VICTIM_BF, ATT_BF = "172.31.69.25", "18.221.219.4"
BOT, C2 = "172.31.69.23", "18.219.211.138"
BG = ["172.31.64.10", "172.31.64.11", "172.31.64.12"]
EXT = "93.184.216.34"


def local(day: str, hhmm: str, sec: float = 0.0) -> float:
    h, m = hhmm.split(":")
    return MIDNIGHT[day] - UTC_OFFSET * 3600 + int(h) * 3600 + int(m) * 60 + sec


def _write(packets) -> bytes:
    from scapy.all import Ether
    from scapy.utils import PcapWriter

    packets = sorted(packets, key=lambda x: float(x.time))
    buf = io.BytesIO()
    w = PcapWriter(buf, sync=True)
    for p in packets:
        frame = Ether() / p
        frame.time = p.time                 # the frame is a new object: carry the intended timestamp over
        w.write(frame)
    w.flush()
    return buf.getvalue()


STRAY = b"leadservices.com\x82\x15googlesyndication.com" * 20          # stray bytes as found in a real capture


def _damage(blob: bytes) -> bytes:
    """The damage found in CSE-CIC-IDS2018 captures: stray bytes between two records, incomplete last record."""
    import struct

    recs, p = [], 24
    while p + 16 <= len(blob):
        incl = struct.unpack_from("<I", blob, p + 8)[0]
        recs.append(blob[p:p + 16 + incl])
        p += 16 + incl
    mid = len(recs) // 2
    return blob[:24] + b"".join(recs[:mid]) + STRAY + b"".join(recs[mid:]) + recs[0][:30]


def _day_archive(day: str, rng) -> tuple[bytes, dict]:
    caps = {ip: [] for ip in BG}
    truth = {"attack_conns": 0, "benign_conns": 0, "shared": 0}
    port = 40000
    # benign: every background host talks to an external web server through the day
    for ip in BG:
        for k in range(25):
            port += 1
            t = local(day, "08:30", rng.uniform(0, 8 * 3600))
            caps[ip] += _tcp_session(t, ip, EXT, port, 443, [b"x" * int(rng.integers(100, 600))], [b"y" * int(rng.integers(300, 1400))])
            truth["benign_conns"] += 1
    # one benign connection between two captured hosts: recorded at both ends
    port += 1
    shared = _tcp_session(local(day, "09:00", 5), BG[0], BG[1], port, 445, [b"m" * 200], [b"n" * 300])
    caps[BG[0]] += shared
    caps[BG[1]] += [p.copy() for p in shared]
    truth["benign_conns"] += 1
    truth["shared"] = 1
    if day == DAY_BF:                       # FTP brute force 10:32-12:09 against the victim
        caps[VICTIM_BF] = []
        for k in range(40):
            port += 1
            t = local(day, "10:40", k * 20.0)
            caps[VICTIM_BF] += _tcp_session(t, ATT_BF, VICTIM_BF, port, 21, [b"USER a\r\n", b"PASS " + TOKEN + b"\r\n"],
                                            [b"331 pw\r\n", b"530 no\r\n"])
            truth["attack_conns"] += 1
        for k in range(10):                 # the victim's own benign traffic
            port += 1
            caps[VICTIM_BF] += _tcp_session(local(day, "09:10", k * 60.0), VICTIM_BF, EXT, port, 443, [b"x" * 300], [b"y" * 900])
            truth["benign_conns"] += 1
    else:                                   # bot calls its command-and-control server 10:11-11:34
        caps[BOT] = []
        for k in range(40):
            port += 1
            t = local(day, "10:20", k * 30.0)
            caps[BOT] += _tcp_session(t, BOT, C2, port, 8080, [b"beacon"], [b"cmd"])
            truth["attack_conns"] += 1
        for k in range(10):
            port += 1
            caps[BOT] += _tcp_session(local(day, "13:00", k * 60.0), BOT, EXT, port, 443, [b"x" * 300], [b"y" * 900])
            truth["benign_conns"] += 1
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (ip, pk) in enumerate(sorted(caps.items())):
            blob = _write(pk)
            if day == DAY_BOT and ip == BOT:          # the bot's capture is damaged in the archive
                import hashlib

                blob = _damage(blob)
                truth["damaged"] = {"capture": f"capHOST-{i:02d}-{ip}", "sha256": hashlib.sha256(blob).hexdigest()}
            zf.writestr(f"pcap/capHOST-{i:02d}-{ip}", blob)
    return buf.getvalue(), truth


@pytest.fixture(scope="module")
def mini_bucket():
    rng = np.random.default_rng(0)
    objects, truth = {}, {}
    for day in (DAY_BF, DAY_BOT):
        blob, tr = _day_archive(day, rng)
        objects[f"Original Network Traffic and Log data/{day}/pcap.zip"] = blob
        truth[day] = tr

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            key = unquote(urlparse(self.path).path.lstrip("/"))
            if key not in objects:
                self.send_error(404)
                return
            blob = objects[key]
            rng_h = self.headers.get("Range")
            a, b = 0, len(blob) - 1
            if rng_h:
                a_s, b_s = rng_h.split("=")[1].split("-")
                a, b = int(a_s), (int(b_s) if b_s else len(blob) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(blob)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            self.wfile.write(blob[a: b + 1])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/", "truth": truth}
    srv.shutdown()


def test_archives_to_corpus(mini_bucket, tmp_path):
    from twws import stages
    from twws.config import apply_dotted_overrides, load_config
    from twws.pipeline import PrepOptions, Preparer

    rule = tmp_path / "test.rules"
    rule.write_text(  # an attack rule, and an informational rule that fires on benign web traffic
        f'alert tcp any any -> any 21 (msg:"TWWS test token"; content:"{TOKEN.decode()}"; classtype:attempted-admin; sid:1000001; rev:1;)\n'
        'alert tcp any any -> any 443 (msg:"TWWS informational"; content:"xxxxxxxxxx"; classtype:misc-activity; sid:1000002; rev:1;)\n')
    cfg = apply_dotted_overrides(load_config(str(REPO / "configs" / "paper.yaml")), [
        f"work_dir={tmp_path / 'work'}", f"corpus.bucket_url={mini_bucket['url']}", f"corpus.days=[{DAY_BF},{DAY_BOT}]",
        "corpus.pinned_inventory=null", "corpus.background_hosts_per_day=all", "corpus.family_cap=1000",
        "corpus.benign_share=null", f"snort.rules=[{rule}]", "extract.n_jobs=2", "snort.n_jobs=2",
        "oracles.enabled=[stat,signature,cluster,domain]", "features.ae_epochs=2", "selection.n_folds=2",
        "oracles.cluster.min_samples=5", "compute.n_jobs=2"])
    rep = stages.stage_ingest(cfg, delete_pcaps=True)
    assert set(rep) == {DAY_BF, DAY_BOT} and all(v.get("captures_deleted") for v in rep.values())
    man = pd.read_csv(tmp_path / "work" / "data" / "manifest.csv")
    assert len(man) == 8 and man["sha256"].str.len().eq(64).all()             # 4 captures per day
    # the damaged capture was repaired before Zeek and Snort; the manifest keeps its digest as distributed
    dmg = mini_bucket["truth"][DAY_BOT]["damaged"]
    chk = pd.read_csv(tmp_path / "work" / "extract" / "capture_checks.csv")
    assert len(chk) == 8 and int(chk["repaired"].sum()) == 1
    row = chk[chk["repaired"].astype(bool)].iloc[0]
    assert row["capture"] == dmg["capture"] and row["stray_regions"] == 1 and row["original_sha256"] == dmg["sha256"]
    assert man.loc[man["member"].str.endswith(dmg["capture"]), "sha256"].item() == dmg["sha256"]
    assert (man.groupby("day")["role"].apply(lambda r: (r == "victim").sum()) == 1).all()
    info = stages.stage_corpus(cfg)
    ws = stages.workspace(cfg)
    sinfo = __import__("json").loads((ws.corpus_dir / "stream_info.json").read_text())
    truth = mini_bucket["truth"]
    n_conn = sum(t["attack_conns"] + t["benign_conns"] for t in truth.values())
    assert sinfo["utc_offset_hours"] == UTC_OFFSET
    diag = {d["utc_offset_hours"]: d["share_inside"] for d in sinfo["utc_offset_diagnostic"]}
    assert diag[UTC_OFFSET] == 1.0 and diag[UTC_OFFSET - 1] < 1.0 and diag[UTC_OFFSET + 1] < 1.0
    assert all(r["inside_window"] == r["flows"] for r in sinfo["label_summary"] if r["family"] in ("BruteForce", "Bot"))
    assert sinfo["deduplication"]["connections_removed"] == sum(t["shared"] for t in truth.values())
    assert sinfo["connections_stream"] == n_conn
    assert sinfo["flows_stream"] == 2 * n_conn                                 # every connection has two directions
    stream = ws.load_stream()
    fam = stream.groupby("family", observed=True)["conn"].nunique().to_dict()
    assert fam["BruteForce"] == truth[DAY_BF]["attack_conns"] and fam["Bot"] == truth[DAY_BOT]["attack_conns"]
    assert fam["Benign"] == sum(t["benign_conns"] for t in truth.values())
    # the attack rule fires on every brute-force connection and on nothing else; the informational rule fires on
    # benign connections, which the first-version flag (every alert) records but the signature oracle ignores
    alert_conns = stream.loc[stream["snort_alert"], ["conn", "family"]].drop_duplicates()
    assert len(alert_conns) == truth[DAY_BF]["attack_conns"] and (alert_conns["family"] == "BruteForce").all()
    any_conns = stream.loc[stream["snort_alert_any"], ["conn", "family"]].drop_duplicates()
    assert len(any_conns) > len(alert_conns) and (any_conns["family"] == "Benign").any()
    assert sinfo["snort_alerts_by_class"].get("misc-activity", 0) > 0 and sinfo["snort_alerts_attack_classes"] > 0
    assert info["selected"]["BruteForce"] == 2 * truth[DAY_BF]["attack_conns"]
    # oracles and aggregation run on the corpus; the signature oracle votes exactly on the alerted flows
    prep = Preparer(ws, cfg, n_jobs=2)
    o = PrepOptions(oracles=("stat", "signature", "cluster", "domain"), cluster_min_samples=5, n_folds=2, latent_dim=8)
    B = prep.stage_b(o)
    assert set(B["votes"]) == {"stat", "signature", "cluster", "domain"}
    assert int((B["votes"]["signature"] == 1).sum()) == int(ws.meta["snort_alert"].sum()) > 0
    assert all(len(v) == len(ws.meta) for v in B["votes"].values())
    # the report builder reads the manifest, the extraction record and the corpus of this workspace:
    # every value that the manuscript takes from the real-data path must be produced
    from twws.report.build import build_report

    status = build_report(cfg, tmp_path / "generated")
    assert status["b_manifest"] == "ok" and status["b_corpus"] == "ok" and status["b_setup"] == "ok", status
    assert status["b_runtime"] == "ok", status
    numbers = __import__("json").loads((tmp_path / "generated" / "numbers.json").read_text())["text"]
    assert numbers["extract.captures_repaired"] == "1" and numbers["extract.captures_checked"] == "8"
    for key in ("manifest.captures", "manifest.victim", "manifest.background", "manifest.gb", "extract.zeek_version",
                "extract.snort_version", "extract.snort_rules", "extract.snort_rule_source", "stream.extracted",
                "stream.dedup_removed", "stream.n", "stream.connections", "stream.utc_offset",
                "stream.offset_share_inside", "stream.n_ambiguous", "corpus.n", "corpus.n_benign",
                "corpus.n_malicious", "corpus.malicious_share", "corpus.n_connections", "corpus.tls_share",
                "corpus.entropy_missing_share", "cfg.corpus.n_days", "feat.d_total"):
        assert key in numbers, key
    assert numbers["manifest.captures"] == "8" and numbers["manifest.victim"] == "2"
    assert numbers["synthetic"] == "0" and numbers["results.available"] == ""
    # a profile that differs only after extraction (configs/paper_reduced.yaml) uses the same captures and
    # Zeek/Snort output through capture_dir, and builds its own corpus in its own work_dir
    cfg_r = apply_dotted_overrides(cfg, [f"work_dir={tmp_path / 'work_reduced'}", f"capture_dir={tmp_path / 'work'}",
                                         "corpus.max_rows=300"])
    info_r = stages.stage_corpus(cfg_r)
    ws_r = stages.workspace(cfg_r)
    assert ws_r.capture_root == ws.root and ws_r.root != ws.root and not (ws_r.root / "extract").exists()
    sinfo_r = __import__("json").loads((ws_r.corpus_dir / "stream_info.json").read_text())
    assert sinfo_r["flows_stream"] == sinfo["flows_stream"] and abs(sum(info_r["selected"].values()) - 300) <= len(info_r["selected"])   # one rounding per class
    status_r = build_report(cfg_r, tmp_path / "generated_r")
    assert status_r["b_manifest"] == "ok" and status_r["b_corpus"] == "ok", status_r
    manifest_tex = (tmp_path / "generated" / "tables" / "manifest.tex").read_text()
    assert "RESULT NOT AVAILABLE" not in manifest_tex and DAY_BF in manifest_tex
    for name in ("manifest.csv", "archive_members.csv", "snort_rule_files.csv", "utc_offset_diagnostic.csv",
                 "label_summary_by_day.csv"):
        assert (tmp_path / "generated" / "supplementary" / name).exists(), name
    supp = (tmp_path / "generated" / "tables" / "supplementary_files.tex").read_text()
    assert "manifest.csv" in supp and "(no description)" not in supp


# --------------------------------------------------------------------------- CICIDS2017: local captures -> transfer model
MIDNIGHT17 = {"Monday": 1499040000.0, "Tuesday": 1499126400.0}     # 00:00 UTC of 2017-07-03 and 2017-07-04
ATT17, VICTIM17, LAN17 = "205.174.165.73", "192.168.10.50", ["192.168.10.5", "192.168.10.8", "192.168.10.9"]


def local17(day: str, hhmm: str, sec: float = 0.0) -> float:
    """Local time of the CICIDS2017 test bed (UTC-3) as epoch seconds."""
    h, m = hhmm.split(":")
    return MIDNIGHT17[day] + 3 * 3600 + int(h) * 3600 + int(m) * 60 + sec


def test_cicids2017_local_captures_to_transfer_model(tmp_path):
    """The CICIDS2017 path of the run script: captures named <Day>-WorkingHours.pcap in one folder (an unfinished
    download in a sub-folder is ignored) -> manifest -> Zeek + Snort -> corpus -> transfer oracle of the main
    configuration, with the configuration values of the paper (worker counts, distribution rules)."""
    from twws import stages
    from twws.config import apply_dotted_overrides, load_config

    rng = np.random.default_rng(1)
    root = tmp_path / "CICIDS2017"
    (root / ".partial").mkdir(parents=True)
    port = 30000
    for day in ("Monday", "Tuesday"):
        pk = []
        for ip in LAN17:                                    # benign traffic of the LAN through the day
            for k in range(20):
                port += 1
                t = local17(day, "08:30", rng.uniform(0, 8 * 3600))
                pk += _tcp_session(t, ip, EXT, port, 443, [b"x" * int(rng.integers(100, 600))],
                                   [b"y" * int(rng.integers(300, 1400))])
        if day == "Tuesday":                                # FTP-Patator 09:20-10:20 from the Kali host
            for k in range(40):
                port += 1
                pk += _tcp_session(local17(day, "09:25", k * 60.0), ATT17, VICTIM17, port, 21,
                                   [b"USER a\r\n", b"PASS b\r\n"], [b"331 pw\r\n", b"530 no\r\n"])
        (root / f"{day}-WorkingHours.pcap").write_bytes(_write(pk))
    (root / ".partial" / "Wednesday-workingHours.pcap").write_bytes(b"\xd4\xc3\xb2\xa1 unfinished")

    src = apply_dotted_overrides(load_config(str(REPO / "configs" / "ids2017.yaml")), [
        f"work_dir={tmp_path / 'work17'}", "corpus.days=[Monday,Tuesday]", "features.ae_epochs=2"])
    f = stages.stage_fetch(src, local_root=str(root))
    assert f["captures"] == 2, f
    stages.stage_extract(src)
    info = stages.stage_corpus(src)
    assert info["n_selected"] > 0
    cfg = apply_dotted_overrides(load_config(str(REPO / "configs" / "paper.yaml")), [
        f"work_dir={tmp_path / 'work18'}", f"transfer.work_dir={tmp_path / 'work17'}"])
    tr = stages.stage_transfer(cfg)
    assert (tmp_path / "work18" / "transfer" / "transfer_rf.joblib").exists()
    # behaviour of the two revised oracle rules on the second corpus (diagnostic only)
    chk = stages.stage_oracle_check(cfg)
    assert {r["quantile"] for r in chk["fanout"]["quantiles"]} >= set(stages.CALIBRATION_QUANTILES)
    assert {"attack_classes", "every_alert_first_version"} <= set(chk["signature"])
    assert (tmp_path / "work17" / "calibration" / "oracle_check.json").exists()
    assert tr["utc_offset_hours"] == -3, tr
    assert tr["families"].get("BruteForce", 0) > 0 and tr["families"].get("Benign", 0) > 0, tr["families"]

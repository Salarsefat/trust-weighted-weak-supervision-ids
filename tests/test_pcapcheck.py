"""Check and repair of damaged capture files (twws.pcapcheck).

The damage reproduced here was found in CSE-CIC-IDS2018 archives: an incomplete last packet record, and
a few hundred stray bytes between two records after which valid records continue.
"""
import json
import struct

import numpy as np
import pytest

from conftest import have

T0 = 1518600000


def _header(snaplen=65535):
    return struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, snaplen, 1)


def _records(n, seed=0, t0=T0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        size = int(rng.integers(60, 1515))
        payload = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        out.append(struct.pack("<IIII", t0 + i // 50, (i % 50) * 20000, size, size) + payload)
    return out


def _split(blob):
    """Header and records of a little-endian classic pcap."""
    recs, p = [], 24
    while p + 16 <= len(blob):
        incl = struct.unpack_from("<I", blob, p + 8)[0]
        recs.append(blob[p:p + 16 + incl])
        p += 16 + incl
    return blob[:24], recs


STRAY_TEXT = b"leadservices.com\x82\x15googlesyndication.com\x82\x15googletagservices.com" * 12   # as in a real capture


def _write(tmp_path, name, blob):
    p = tmp_path / name
    p.write_bytes(blob)
    return p


def test_complete_capture_is_left_unchanged(tmp_path):
    from twws.pcapcheck import check_and_repair, sidecar

    blob = _header() + b"".join(_records(300))
    p = _write(tmp_path, "clean.pcap", blob)
    info = check_and_repair(p)
    assert info["status"] == "complete" and info["records"] == 300 and not info["repaired"] and not info["repaired_now"]
    assert p.read_bytes() == blob and json.loads(sidecar(p).read_text())["final_size"] == len(blob)


@pytest.mark.parametrize("damage", ["tail", "stray", "stray+tail", "stray at end"])
def test_damaged_capture_is_repaired_to_its_valid_records(tmp_path, damage):
    from twws.pcapcheck import check_and_repair

    recs = _records(400, seed=1)
    clean = _header() + b"".join(recs)
    parts = list(recs)
    stray = 0
    if "stray" in damage and damage != "stray at end":
        parts.insert(120, STRAY_TEXT[:782])
        parts.insert(300, STRAY_TEXT[:17])
        stray = 782 + 17
    blob = _header() + b"".join(parts)
    if damage == "stray at end":
        blob += STRAY_TEXT[:500]
        stray = 500
    if "tail" in damage:
        blob += recs[5][:16 + 57]                      # header of a 1000-byte record, only 57 bytes of it
    p = _write(tmp_path, "damaged.pcap", blob)
    info = check_and_repair(p)
    assert info["repaired"] and info["repaired_now"]
    assert info["records"] == 400
    assert p.read_bytes() == clean                     # exactly the complete records remain, in order
    assert info["removed_bytes"] == len(blob) - len(clean)
    if "stray" in damage and damage != "stray at end":
        assert info["status"] == "stray bytes between records" and info["stray_regions"] == 2
    again = check_and_repair(p)
    assert not again["repaired_now"] and again["final_size"] == len(clean)


def test_repair_across_read_windows(tmp_path, monkeypatch):
    """Damaged regions near the boundaries of the 64 MiB read windows (made small here)."""
    import twws.pcapcheck as pc

    monkeypatch.setattr(pc, "CHUNK", 1 << 16)
    recs = _records(4000, seed=2)                      # about 3 MB: many windows of 64 KiB plus 1 MiB look-ahead
    clean = _header() + b"".join(recs)
    parts = list(recs)
    for i in (2900, 1800, 700, 90):                    # descending, so that the indices stay valid
        parts.insert(i, STRAY_TEXT[: 100 + i % 900])
    blob = _header() + b"".join(parts) + recs[0][:20]
    p = _write(tmp_path, "big.pcap", blob)
    info = pc.check_and_repair(p)
    assert info["records"] == 4000 and info["stray_regions"] == 4
    assert p.read_bytes() == clean


def test_scan_digest_and_non_capture(tmp_path):
    import hashlib

    from twws.pcapcheck import check_and_repair, scan_capture

    blob = _header() + b"".join(_records(50)) + STRAY_TEXT[:300]
    p = _write(tmp_path, "x.pcap", blob)
    assert scan_capture(p)["sha256"] == hashlib.sha256(blob).hexdigest()
    q = _write(tmp_path, "page.html", b"<html>download form</html>" * 10)
    info = check_and_repair(q)
    assert info["status"] == "not a capture" and not info["repaired"] and q.read_bytes().startswith(b"<html>")


def test_pcapng_with_incomplete_last_block(tmp_path):
    from twws.pcapcheck import check_and_repair

    def block(btype, body):
        body += b"\0" * (-len(body) % 4)
        tl = 12 + len(body)
        return struct.pack("<II", btype, tl) + body + struct.pack("<I", tl)

    shb = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    idb = block(1, struct.pack("<HHI", 1, 0, 65535))
    epbs = [block(6, struct.pack("<IIIII", 0, 0, i, 60, 60) + bytes(60)) for i in range(20)]
    clean = shb + idb + b"".join(epbs)
    p = _write(tmp_path, "x.pcapng", clean + epbs[0][:40])
    info = check_and_repair(p)
    assert info["format"] == "pcapng" and info["records"] == 22 and info["repaired"]
    assert p.read_bytes() == clean


@pytest.mark.skipif(not have("zeek"), reason="zeek not installed")
def test_zeek_reads_every_connection_of_a_repaired_capture(tmp_path):
    """Zeek stops with a fatal error at the damage and loses the open connections; after the repair it
    exits normally and reports the same connections as on the undamaged capture."""
    from scapy.all import IP, Ether, wrpcap

    from test_tools import _tcp_session
    from twws.extract import flows_from_log, run_zeek, zeek_binary
    from twws.pcapcheck import check_and_repair

    pk = []
    for k in range(30):
        pk += _tcp_session(T0 + k * 2.0, "172.31.64.10", "172.31.69.25", 41000 + k, 443, [b"a" * 300], [b"b" * 900])
    pk.sort(key=lambda x: float(x.time))
    frames = []
    for x in pk:
        f = Ether() / x
        f.time = x.time
        frames.append(f)
    clean_p = tmp_path / "clean.pcap"
    wrpcap(str(clean_p), frames)
    head, recs = _split(clean_p.read_bytes())
    damaged = head + b"".join(recs[:150]) + STRAY_TEXT[:782] + b"".join(recs[150:]) + recs[3][:30]
    bad_p = _write(tmp_path, "damaged.pcap", damaged)
    with pytest.raises(RuntimeError, match="exit code"):
        run_zeek(bad_p, tmp_path / "z_bad", zeek_binary())
    info = check_and_repair(bad_p)
    assert info["repaired"] and bad_p.read_bytes() == clean_p.read_bytes()
    n_clean = len(flows_from_log(run_zeek(clean_p, tmp_path / "z_clean", zeek_binary())))
    n_rep = len(flows_from_log(run_zeek(bad_p, tmp_path / "z_rep", zeek_binary())))
    assert n_rep == n_clean == 60

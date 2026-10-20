"""Ranged retrieval of single captures from a ZIP archive, selection rule and manifest.

A local HTTP server that honours ``Range`` requests stands in for the public bucket.
"""
import hashlib
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pytest

from twws import fetch
from twws.config import Cfg
from twws.utils import s3_multipart_etag, sha256_file

from conftest import SCHEDULE

DAYS = ["Wednesday-14-02-2018", "Friday-02-03-2018"]
VICTIMS = {"Wednesday-14-02-2018": ["172.31.69.25"],
           "Friday-02-03-2018": ["172.31.69.23", "172.31.69.17", "172.31.69.14", "172.31.69.12", "172.31.69.10",
                                 "172.31.69.8", "172.31.69.6", "172.31.69.26", "172.31.69.29", "172.31.69.30"]}


def _archive(day: str, seed: int) -> tuple[bytes, dict]:
    rng = np.random.default_rng(seed)
    hosts = VICTIMS[day][:2] + [f"172.31.64.{i}" for i in range(10, 22)]
    buf, content = io.BytesIO(), {}
    with zipfile.ZipFile(buf, "w") as zf:
        for i, ip in enumerate(hosts):
            name = f"pcap/capDESKTOP-{i:03d}-{ip}"
            data = rng.integers(0, 256, size=int(rng.integers(20_000, 120_000)), dtype=np.uint8).tobytes()
            zf.writestr(zipfile.ZipInfo(name), data,
                        compress_type=zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED)
            content[name] = data
        zf.writestr("pcap/readme.txt", "not a capture")
    return buf.getvalue(), content


@pytest.fixture(scope="module")
def bucket():
    objects, truth = {}, {}
    for i, day in enumerate(DAYS):
        blob, content = _archive(day, i)
        objects[f"{fetch.IDS2018_PCAP_PREFIX}{day}/pcap.zip"] = blob
        truth[day] = content
    stats = {"bytes": 0, "full_gets": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            key = unquote(u.path.lstrip("/"))
            if key == "":                                    # bucket listing (S3 ListObjectsV2)
                items = "".join(f"<Contents><Key>{k}</Key><Size>{len(v)}</Size><ETag>&quot;{hashlib.md5(v).hexdigest()}&quot;</ETag>"
                                f"<LastModified>2018-01-01T00:00:00.000Z</LastModified></Contents>" for k, v in objects.items())
                body = (f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                        f"<IsTruncated>false</IsTruncated>{items}</ListBucketResult>").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if key not in objects:
                self.send_error(404)
                return
            blob = objects[key]
            rng = self.headers.get("Range")
            if rng:
                a, b = rng.split("=")[1].split("-")
                a = int(a)
                b = int(b) if b else len(blob) - 1
                part = blob[a: b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(blob)}")
            else:
                part = blob
                stats["full_gets"] += 1
                self.send_response(200)
            stats["bytes"] += len(part)
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/", "objects": objects, "truth": truth, "stats": stats}
    srv.shutdown()


def _cfg(bucket_url: str) -> Cfg:
    return Cfg({"corpus": Cfg({"schedule": SCHEDULE, "bucket_url": bucket_url, "days": DAYS, "seed": 42,
                               "background_hosts_per_day": 3, "exclude_hosts": [], "range_block_mb": 1,
                               "max_background_member_bytes": None})})


def test_range_reader_lists_and_extracts_single_members(bucket, tmp_path):
    key = f"{fetch.IDS2018_PCAP_PREFIX}{DAYS[0]}/pcap.zip"
    f = fetch.HTTPRangeFile(fetch.object_url(bucket["url"], key), block_size=16 * 1024)
    members = fetch.list_zip_members(f)
    assert len(members) == len(bucket["truth"][DAYS[0]]) + 1
    name = sorted(bucket["truth"][DAYS[0]])[3]
    info = fetch.extract_member(f, name, tmp_path / "one.pcap")
    data = bucket["truth"][DAYS[0]][name]
    assert (tmp_path / "one.pcap").read_bytes() == data
    assert info["file_size"] == len(data)
    assert f.bytes_fetched < len(bucket["objects"][key])        # only part of the archive was transferred


def test_selection_is_deterministic_and_keeps_every_victim():
    names = [f"pcap/cap-{ip}" for ip in ["172.31.69.25"] + [f"172.31.64.{i}" for i in range(10, 40)]]
    members = pd.DataFrame({"member": names, "file_size": 1000})
    a = fetch.select_members(members, ["172.31.69.25"], 5, seed=7)
    b = fetch.select_members(members, ["172.31.69.25"], 5, seed=7)
    c = fetch.select_members(members, ["172.31.69.25"], 5, seed=8)
    assert a["role"].tolist() == b["role"].tolist()
    assert a["role"].tolist() != c["role"].tolist()
    assert (a.loc[a["host_ip"] == "172.31.69.25", "role"] == "victim").all()
    assert (a["role"] == "background").sum() == 5
    everything = fetch.select_members(members, [], "all", seed=0)
    assert (everything["role"] == "background").all()


def test_fetch_writes_a_file_level_manifest(bucket, tmp_path):
    cfg = _cfg(bucket["url"])
    listed = fetch.fetch_ids2018(cfg, tmp_path / "list", list_only=True)
    assert not (tmp_path / "list" / "manifest.csv").exists()
    assert set(listed.members["role"]) == {"victim", "background", "unselected"}
    before = bucket["stats"]["full_gets"]
    res = fetch.fetch_ids2018(cfg, tmp_path / "data")
    assert bucket["stats"]["full_gets"] == before               # nothing was downloaded as a whole
    man = pd.read_csv(tmp_path / "data" / "manifest.csv")
    assert list(man.columns) == fetch.MANIFEST_COLUMNS
    # per day: the captures of the victims present in the archive plus three background captures
    assert len(man) == sum(min(2, len(VICTIMS[d])) + 3 for d in DAYS)
    assert (man.groupby("day")["role"].apply(lambda r: (r == "background").sum()) == 3).all()
    for _, m in man.iterrows():
        data = bucket["truth"][m.day][m.member]
        local = tmp_path / "data" / m.local_path
        assert local.read_bytes() == data
        assert m.sha256 == hashlib.sha256(data).hexdigest() == sha256_file(local)
        assert m.file_size == len(data)
    # one more call for a single day leaves the other day's rows in place
    fetch.fetch_ids2018(cfg, tmp_path / "data", days=[DAYS[0]])
    assert len(pd.read_csv(tmp_path / "data" / "manifest.csv")) == len(man)


def test_local_archives_give_the_same_retained_set(bucket, tmp_path):
    cfg = _cfg(bucket["url"])
    remote = fetch.fetch_ids2018(cfg, tmp_path / "remote")
    for day in DAYS:
        d = tmp_path / "archives" / day
        d.mkdir(parents=True)
        (d / "pcap.zip").write_bytes(bucket["objects"][f"{fetch.IDS2018_PCAP_PREFIX}{day}/pcap.zip"])
    local = fetch.fetch_ids2018(cfg, tmp_path / "local", local_archives=tmp_path / "archives")
    assert sorted(remote.manifest["sha256"]) == sorted(local.manifest["sha256"])


def test_multipart_etag(tmp_path):
    data = bytes(range(256)) * 1000
    p = tmp_path / "blob"
    p.write_bytes(data)
    part = 100_000
    digests = [hashlib.md5(data[i: i + part]).digest() for i in range(0, len(data), part)]
    expected = hashlib.md5(b"".join(digests)).hexdigest() + f"-{len(digests)}"
    assert s3_multipart_etag(p, part) == expected

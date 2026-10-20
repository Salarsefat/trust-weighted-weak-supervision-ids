"""Capture retrieval with a file-level manifest and checksums.

CSE-CIC-IDS2018 distributes one archive of per-host captures per day
(``Original Network Traffic and Log data/<day>/pcap.zip``, 38-59 GB each; one
day, Tuesday-20-02-2018, is a RAR archive).  ZIP archives carry their directory
at the end of the file, so individual captures can be listed and retrieved with
HTTP range requests without downloading the whole archive.  This module

* reads the public bucket listing and compares it with the pinned inventory,
* lists every member of each daily archive (name, sizes, CRC-32),
* selects members by an explicit, seeded rule (victim hosts + background hosts),
* downloads the selected members, verifies size and CRC-32, and
* writes ``manifest.csv`` with a SHA-256 digest per retained capture.

No AWS account is needed; the bucket is public.
"""
from __future__ import annotations

import csv
import io
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .utils import ensure_dir, get_logger, sha256_file

try:  # ZIP method 9 (Deflate64) is used by some Windows tools for members > 2 GiB
    import zipfile_deflate64  # noqa: F401  (import patches ``zipfile``)

    HAVE_DEFLATE64 = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_DEFLATE64 = False

log = get_logger("twws.fetch")

IDS2018_BUCKET_URL = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
IDS2018_PCAP_PREFIX = "Original Network Traffic and Log data/"
IP_RE = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d{1,3}){3})(?!\d)")


# --------------------------------------------------------------------------- bucket listing
def list_bucket(bucket_url: str = IDS2018_BUCKET_URL, session: requests.Session | None = None) -> pd.DataFrame:
    """Return the public S3 listing as a table (key, size, etag, last_modified)."""
    sess = session or requests.Session()
    rows, token = [], None
    while True:
        params = {"list-type": "2", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = sess.get(bucket_url, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        pref = "s3:" if ns else ""
        for c in root.findall(f"{pref}Contents", ns):
            rows.append({
                "key": c.findtext(f"{pref}Key", namespaces=ns),
                "size": int(c.findtext(f"{pref}Size", namespaces=ns)),
                "etag": (c.findtext(f"{pref}ETag", namespaces=ns) or "").strip('"'),
                "last_modified": c.findtext(f"{pref}LastModified", namespaces=ns),
            })
        if (root.findtext(f"{pref}IsTruncated", namespaces=ns) or "false").lower() != "true":
            break
        token = root.findtext(f"{pref}NextContinuationToken", namespaces=ns)
    return pd.DataFrame(rows)


def compare_inventory(live: pd.DataFrame, pinned_csv: str | os.PathLike) -> pd.DataFrame:
    """Join the live listing with the pinned inventory; flag size/ETag differences."""
    pinned = pd.read_csv(pinned_csv)
    m = pinned.merge(live, on="key", how="outer", suffixes=("_pinned", "_live"), indicator=True)
    m["size_match"] = m["size_pinned"] == m["size_live"]
    m["etag_match"] = m["etag_pinned"] == m["etag_live"]
    return m


# --------------------------------------------------------------------------- HTTP range reader
class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only file object backed by HTTP range requests with a block cache."""

    def __init__(self, url: str, session: requests.Session | None = None, block_size: int = 8 << 20,
                 cache_blocks: int = 6, retries: int = 6, timeout: float = 120.0) -> None:
        super().__init__()
        self.url = url
        self.sess = session or requests.Session()
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.retries = retries
        self.timeout = timeout
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self.bytes_fetched = 0
        self.requests_made = 0
        self._size = self._probe_size()

    # -- size discovery
    def _probe_size(self) -> int:
        r = self._request({"Range": "bytes=0-0"})
        cr = r.headers.get("Content-Range")
        if r.status_code == 206 and cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        if r.status_code == 200 and "Content-Length" in r.headers:  # server ignored the range
            raise OSError(f"server does not honour HTTP range requests for {self.url}")
        raise OSError(f"cannot determine size of {self.url} (HTTP {r.status_code})")

    def _request(self, headers: dict) -> requests.Response:
        last = None
        for attempt in range(self.retries):
            try:
                r = self.sess.get(self.url, headers=headers, timeout=self.timeout)
                if r.status_code in (200, 206):
                    self.requests_made += 1
                    return r
                last = OSError(f"HTTP {r.status_code} for {self.url}")
                if r.status_code in (403, 404):
                    break
            except requests.RequestException as exc:  # network hiccup: back off and retry
                last = exc
            time.sleep(min(2 ** attempt, 30))
        raise OSError(f"range request failed after {self.retries} attempts: {last}")

    def _block(self, idx: int) -> bytes:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        start = idx * self.block_size
        end = min(start + self.block_size, self._size) - 1
        r = self._request({"Range": f"bytes={start}-{end}"})
        data = r.content
        if len(data) != end - start + 1:
            raise OSError(f"short read: wanted {end - start + 1} bytes, got {len(data)}")
        self.bytes_fetched += len(data)
        self._cache[idx] = data
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    # -- io.RawIOBase interface
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:  # pragma: no cover
            raise ValueError("invalid whence")
        self._pos = max(0, self._pos)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = max(0, min(n, self._size - self._pos))
        if n == 0:
            return b""
        out = bytearray()
        pos = self._pos
        while len(out) < n:
            idx, off = divmod(pos, self.block_size)
            chunk = self._block(idx)[off: off + (n - len(out))]
            if not chunk:
                break
            out += chunk
            pos += len(chunk)
        self._pos = pos
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    @property
    def size(self) -> int:
        return self._size


def object_url(bucket_url: str, key: str) -> str:
    return bucket_url.rstrip("/") + "/" + quote(key)


# --------------------------------------------------------------------------- ZIP members
METHOD_NAMES = {0: "stored", 8: "deflate", 9: "deflate64", 12: "bzip2", 14: "lzma"}


def list_zip_members(fileobj) -> pd.DataFrame:
    """List the members of a ZIP archive opened on any seekable file object."""
    with zipfile.ZipFile(fileobj) as zf:
        rows = []
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            rows.append({
                "member": zi.filename,
                "file_size": zi.file_size,
                "compress_size": zi.compress_size,
                "crc32": f"{zi.CRC & 0xFFFFFFFF:08x}",
                "method": METHOD_NAMES.get(zi.compress_type, str(zi.compress_type)),
                "header_offset": zi.header_offset,
            })
    return pd.DataFrame(rows)


def extract_member(fileobj, member: str, dest: str | os.PathLike, chunk: int = 1 << 20) -> dict:
    """Stream one member to ``dest``; the ZIP layer verifies CRC-32 at end of stream."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with zipfile.ZipFile(fileobj) as zf:
        zi = zf.getinfo(member)
        if zi.compress_type == 9 and not HAVE_DEFLATE64:
            raise RuntimeError("member uses Deflate64; install the 'zipfile-deflate64' package")
        n = 0
        with zf.open(zi) as src, open(tmp, "wb") as dst:
            while True:
                buf = src.read(chunk)
                if not buf:
                    break
                dst.write(buf)
                n += len(buf)
    if n != zi.file_size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"size mismatch for {member}: got {n}, expected {zi.file_size}")
    os.replace(tmp, dest)
    return {"member": member, "file_size": n, "crc32": f"{zi.CRC & 0xFFFFFFFF:08x}"}


# --------------------------------------------------------------------------- selection rule
def member_ip(name: str) -> str | None:
    m = IP_RE.search(os.path.basename(name))
    return m.group(1) if m else None


def select_members(members: pd.DataFrame, victim_ips: list[str], n_background: int | str, seed: int,
                   exclude_ips: list[str] | None = None, max_member_bytes: int | None = None) -> pd.DataFrame:
    """Deterministic member selection for one day.

    * every capture whose file name carries a victim IP of that day is retained (role ``victim``);
    * ``n_background`` further captures are drawn uniformly at random, without replacement,
      from the remaining captures sorted by name (role ``background``); ``"all"`` keeps all;
    * captures above ``max_member_bytes`` (uncompressed) are never drawn as background.

    The rule depends only on the archive listing, the attack schedule and ``seed``.
    """
    df = members.copy()
    df["host_ip"] = df["member"].map(member_ip)
    df["role"] = "unselected"
    vic = set(victim_ips or [])
    df.loc[df["host_ip"].isin(vic), "role"] = "victim"
    keep_all = isinstance(n_background, str) and n_background.lower() == "all"
    pool = df[df["role"] == "unselected"]
    if not keep_all:                      # a random draw needs the host address in the file name
        pool = pool[pool["host_ip"].notna()]
    if exclude_ips:
        pool = pool[~pool["host_ip"].isin(set(exclude_ips))]
    if max_member_bytes and not keep_all:
        pool = pool[pool["file_size"] <= max_member_bytes]
    pool = pool.sort_values("member")
    if keep_all:
        chosen = pool.index
    else:
        k = min(int(n_background), len(pool))
        rng = np.random.default_rng(seed)
        chosen = pool.index[np.sort(rng.choice(len(pool), size=k, replace=False))] if k > 0 else []
    df.loc[chosen, "role"] = "background"
    return df


# --------------------------------------------------------------------------- generic download
def download_file(url: str, dest: str | os.PathLike, session: requests.Session | None = None,
                  chunk: int = 1 << 22, retries: int = 8) -> Path:
    """Resumable download of a whole object (used for the RAR day and auxiliary files)."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    sess = session or requests.Session()
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:  # already complete
                    break
                if have and r.status_code == 200:  # server restarted from zero
                    have = 0
                    tmp.unlink(missing_ok=True)
                r.raise_for_status()
                with open(tmp, "ab" if have else "wb") as fh:
                    for buf in r.iter_content(chunk):
                        fh.write(buf)
            break
        except requests.RequestException as exc:
            log.warning("download interrupted (%s); retry %d/%d", exc, attempt + 1, retries)
            time.sleep(min(2 ** attempt, 60))
    else:
        raise OSError(f"could not download {url}")
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- manifest
MANIFEST_COLUMNS = ["dataset", "day", "archive_key", "archive_size", "archive_etag", "member", "role", "host_ip",
                    "file_size", "crc32", "sha256", "local_path"]


@dataclass
class FetchResult:
    members: pd.DataFrame   # every member of every listed archive, with the selection role
    manifest: pd.DataFrame  # retained captures only, with SHA-256


def _merge_csv(path: Path, new: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """Replace the rows of ``days`` in an existing CSV by ``new`` (so that days can be fetched one by one)."""
    if path.exists():
        try:
            old = pd.read_csv(path)
            if "day" in old.columns:
                new = pd.concat([old[~old["day"].isin(days)], new], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    return new


def find_local_archive(root: str | os.PathLike, day: str, name: str = "pcap.zip") -> Path | None:
    """Locate a daily archive that was downloaded beforehand (several common layouts)."""
    root = Path(root)
    for cand in (root / day / name, root / IDS2018_PCAP_PREFIX.rstrip("/") / day / name, root / f"{day}.zip",
                 root / f"{day}-{name}"):
        if cand.is_file():
            return cand
    return None


def fetch_ids2018(cfg, out_dir: str | os.PathLike, list_only: bool = False, session: requests.Session | None = None,
                  days: list[str] | None = None, local_archives: str | os.PathLike | None = None) -> FetchResult:
    """List, select and (unless ``list_only``) retrieve the retained CSE-CIC-IDS2018 captures.

    The daily archives are read with HTTP range requests from the public bucket, or from disk
    when ``local_archives`` points at a folder that already holds them.  ``days`` restricts the
    call to some capture days; ``manifest.csv`` and ``archive_members.csv`` are updated in place,
    so that days can be processed one after the other.
    """
    from .labels import load_schedule

    out_dir = ensure_dir(out_dir)
    sched = load_schedule(cfg.corpus.schedule)
    sess = session or requests.Session()
    bucket = cfg.corpus.get("bucket_url", IDS2018_BUCKET_URL)
    days = list(days or cfg.corpus.days)
    inv = None
    if local_archives is None:
        try:
            inv = list_bucket(bucket, sess)
            inv.to_csv(out_dir / "bucket_listing_live.csv", index=False)
        except Exception as exc:  # the listing is informative only
            log.warning("bucket listing unavailable: %s", exc)

    all_members, manifest_rows = [], []
    for day in days:
        dinfo = sched["days"][day]
        key = f"{IDS2018_PCAP_PREFIX}{day}/{dinfo.get('archive', 'pcap.zip')}"
        if not key.lower().endswith(".zip"):
            log.warning("%s is not a ZIP archive (%s); it needs a full download - see docs/DATA.md", day, key)
            continue
        etag = size = None
        if local_archives is not None:
            path = find_local_archive(local_archives, day, os.path.basename(key))
            if path is None:
                log.warning("no local archive for %s under %s", day, local_archives)
                continue
            fobj = open(path, "rb")
            size = path.stat().st_size
        else:
            fobj = HTTPRangeFile(object_url(bucket, key), session=sess,
                                 block_size=int(cfg.corpus.get("range_block_mb", 8)) << 20)
            if inv is not None and (inv.key == key).any():
                row = inv[inv.key == key].iloc[0]
                etag, size = row.etag, int(row["size"])
        try:
            members = list_zip_members(fobj)
            victims = dinfo.get("victims_internal", [])
            sel = select_members(members, victims, cfg.corpus.background_hosts_per_day,
                                 seed=int(cfg.corpus.seed) + sorted(sched["days"]).index(day),
                                 exclude_ips=cfg.corpus.get("exclude_hosts"),
                                 max_member_bytes=cfg.corpus.get("max_background_member_bytes"))
            sel.insert(0, "day", day)
            sel.insert(1, "archive_key", key)
            all_members.append(sel)
            log.info("%s: %d members, %d victim + %d background retained (%.1f GiB uncompressed)", day, len(sel),
                     (sel.role == "victim").sum(), (sel.role == "background").sum(),
                     sel.loc[sel.role != "unselected", "file_size"].sum() / 2**30)
            missing_victims = sorted(set(victims) - set(sel.loc[sel.role == "victim", "host_ip"]))
            if missing_victims:
                log.warning("%s: no capture found for victim host(s) %s", day, missing_victims)
            if list_only:
                continue
            for _, m in sel[sel.role != "unselected"].iterrows():
                local = out_dir / "pcap" / day / os.path.basename(m.member)
                if not (local.exists() and local.stat().st_size == m.file_size) and not _repaired_copy(local, m.file_size):
                    log.info("retrieving %s (%.0f MiB)", m.member, m.file_size / 2**20)
                    extract_member(fobj, m.member, local)
                manifest_rows.append({
                    "dataset": "CSE-CIC-IDS2018", "day": day, "archive_key": key, "archive_size": size,
                    "archive_etag": etag, "member": m.member, "role": m.role, "host_ip": m.host_ip,
                    "file_size": int(m.file_size), "crc32": m.crc32, "sha256": _original_sha256(local),
                    "local_path": str(local.relative_to(out_dir)),
                })
        finally:
            fobj.close()
    members_df = pd.concat(all_members, ignore_index=True) if all_members else pd.DataFrame()
    if len(members_df):
        members_df = _merge_csv(out_dir / "archive_members.csv", members_df, days)
        members_df.to_csv(out_dir / "archive_members.csv", index=False)
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    if not list_only:
        manifest = _merge_csv(out_dir / "manifest.csv", manifest, days)
        manifest.to_csv(out_dir / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return FetchResult(members_df, manifest)


def _repaired_copy(local: Path, archive_size: int) -> bool:
    """True if ``local`` is the repaired form of the archive member of this size (see ``twws.pcapcheck``)."""
    from .pcapcheck import stored_check

    if not local.exists():
        return False
    chk = stored_check(local)
    return bool(chk and chk.get("original_size") == archive_size and chk.get("final_size") == local.stat().st_size)


def _original_sha256(local: Path) -> str:
    """SHA-256 of the capture as distributed (before any repair)."""
    from .pcapcheck import stored_check

    chk = stored_check(local)
    if chk and chk.get("final_size") == local.stat().st_size and chk.get("original_sha256"):
        return chk["original_sha256"]
    return sha256_file(local)


def _original_size(local: Path) -> int:
    from .pcapcheck import stored_check

    chk = stored_check(local)
    if chk and chk.get("final_size") == local.stat().st_size and chk.get("original_size"):
        return int(chk["original_size"])
    return int(local.stat().st_size)


def manifest_from_local(cfg, pcap_root: str | os.PathLike, out_dir: str | os.PathLike) -> pd.DataFrame:
    """Build the manifest for captures that are already on disk (``<root>/<day>/<capture>``).

    The same selection rule is applied to the directory listing, so that a local copy of the
    full archives and a ranged retrieval produce the same retained set.
    """
    from .labels import load_schedule

    out_dir = ensure_dir(out_dir)
    sched = load_schedule(cfg.corpus.schedule)
    rows, listed = [], []
    # not captures: notes, sidecars, and files that a download or a repair is still writing
    skip = (".part", ".log", ".csv", ".zip", ".rar", ".txt", ".md", ".json", ".repairing", ".not-a-capture",
            ".crdownload", ".partial", ".tmp", ".download", ".opdownload", ".md5", ".md5-mismatch")
    for day in cfg.corpus.days:
        ddir = Path(pcap_root) / day
        if ddir.is_dir():
            files = sorted(p for p in ddir.rglob("*") if p.is_file() and not p.name.lower().endswith(skip))
        else:   # flat layout: captures named after the day, e.g. Monday-WorkingHours.pcap
            ddir = Path(pcap_root)
            files = sorted(p for p in ddir.iterdir() if p.is_file() and p.name.lower().startswith(day.lower())
                           and not p.name.lower().endswith(skip))
        if not files:
            log.warning("no local captures for %s under %s", day, pcap_root)
            continue
        members = pd.DataFrame({"member": [str(p.relative_to(ddir)) for p in files],
                                "file_size": [_original_size(p) for p in files]})
        members["compress_size"] = np.nan
        members["crc32"] = ""
        sel = select_members(members, sched["days"][day].get("victims_internal", []),
                             cfg.corpus.background_hosts_per_day,
                             seed=int(cfg.corpus.seed) + sorted(sched["days"]).index(day),
                             exclude_ips=cfg.corpus.get("exclude_hosts"),
                             max_member_bytes=cfg.corpus.get("max_background_member_bytes"))
        sel.insert(0, "day", day)
        listed.append(sel)
        for _, m in sel[sel.role != "unselected"].iterrows():
            p = ddir / m.member
            rows.append({"dataset": cfg.corpus.get("name", "CSE-CIC-IDS2018"), "day": day, "archive_key": "",
                         "archive_size": None, "archive_etag": None, "member": m.member, "role": m.role,
                         "host_ip": m.host_ip, "file_size": int(m.file_size), "crc32": "",
                         "sha256": _original_sha256(p), "local_path": str(p)})
    if listed:
        pd.concat(listed, ignore_index=True).to_csv(out_dir / "archive_members.csv", index=False)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    log.info("manifest of %d local captures written (%.1f GiB)", len(manifest), manifest["file_size"].sum() / 2**30)
    return manifest

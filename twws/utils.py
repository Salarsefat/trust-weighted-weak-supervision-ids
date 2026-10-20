"""Shared helpers: logging, timing, seeding, hashing, JSON output and environment capture."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str = "twws") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("TWWS_LOGLEVEL", "INFO"), format=LOG_FORMAT, stream=sys.stderr)
    return logger


log = get_logger()


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if present) PyTorch; return a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # torch is optional for most of the pipeline
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch not installed
        pass
    return np.random.default_rng(seed)


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux reports kilobytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # bytes on macOS
        return rss / (1024 * 1024)
    return rss / 1024


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover
        return float("nan")


def release_memory() -> None:
    """Return freed memory to the operating system: garbage collector, Arrow's memory pool, C heap.

    The large stages allocate through two allocators (Arrow for strings and Parquet, the C heap for NumPy);
    memory that one of them keeps after a step cannot be reused by the other, so it is released between steps.
    """
    import gc

    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:  # pragma: no cover
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # pragma: no cover
            pass


class StageTimer:
    """Collects wall-clock time and peak memory per named pipeline stage."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        rss0 = current_rss_mb()
        log.info("stage start: %s", name)
        try:
            yield
        finally:
            release_memory()
            dt = time.perf_counter() - t0
            rec = {
                "stage": name,
                "seconds": round(dt, 3),
                "rss_start_mb": round(rss0, 1),
                "rss_end_mb": round(current_rss_mb(), 1),
                "peak_rss_mb_process": round(peak_rss_mb(), 1),
            }
            self.records.append(rec)
            log.info("stage done: %s (%.1f s, rss %.0f MiB)", name, dt, rec["rss_end_mb"])

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.records)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def s3_multipart_etag(path: str | os.PathLike, part_size: int) -> str:
    """ETag S3 reports for a multipart upload with fixed ``part_size`` bytes per part.

    The ETag is ``md5(concat(md5(part_i))) + '-' + n_parts``; for a single-part
    upload it is the plain MD5 of the object.
    """
    digests = []
    with open(path, "rb") as fh:
        while True:
            b = fh.read(part_size)
            if not b:
                break
            digests.append(hashlib.md5(b).digest())
    if len(digests) == 1:
        return digests[0].hex()
    return hashlib.md5(b"".join(digests)).hexdigest() + f"-{len(digests)}"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401 - json hook
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean(obj):
    """Replace NaN/inf floats by None so that the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, indent=2, cls=_NumpyEncoder, sort_keys=False)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(cmd: list[str], cwd: str | None = None, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout)


def tool_version(binary: str, args: tuple[str, ...] = ("--version",)) -> str | None:
    """First output line of ``binary --version`` that carries a version number (None if not installed)."""
    exe = binary if os.path.isabs(binary) and os.path.exists(binary) else shutil.which(binary)
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=20)
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            m = re.search(r"[A-Za-z/][\w() .,/+-]*\d+\.\d+[\w() .,/+-]*", line)
            if m:
                words = m.group(0).strip().split(" ")
                words[0] = os.path.basename(words[0])       # "/opt/zeek/bin/zeek version 8.0.10" -> "zeek version 8.0.10"
                return " ".join(words)
        return text.strip()[:200] or None
    except Exception:  # pragma: no cover
        return None


def _find_tool(name: str, extra_dirs: tuple[str, ...] = ()) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    for d in extra_dirs:
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
    return name


def environment_record(extra_tools: dict[str, str] | None = None) -> dict[str, Any]:
    """Versions of the interpreter, key packages and external tools, plus hardware."""
    import importlib

    pkgs = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "torch", "optuna",
                 "numba", "pyarrow", "pytorch_tabnet", "boruta", "matplotlib"]:
        try:
            pkgs[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception:
            pkgs[name] = None
    cpu_model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = platform.processor() or None
    mem_gb = None
    try:
        import psutil

        mem_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # pragma: no cover
        pass
    tools = {
        "zeek": tool_version(_find_tool("zeek", ("/opt/zeek/bin", "/usr/local/zeek/bin"))),
        "snort": tool_version(_find_tool("snort", ("/usr/sbin", "/usr/local/bin")), ("-V",)),
        "tshark": tool_version("tshark"),
    }
    for k, v in (extra_tools or {}).items():
        tools[k] = v
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_gb": mem_gb,
        "packages": pkgs,
        "tools": tools,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def worker_count(value: Any) -> int:
    """Worker processes for a configured value.

    A positive number is used as given; ``null`` (None), 0 or a negative number means all logical CPUs but one.
    """
    try:
        n = int(value) if value is not None else 0
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else max(1, (os.cpu_count() or 2) - 1)


def stable_id(*parts: str) -> int:
    """Positive 31-bit integer derived from the given strings (stable across runs and machines)."""
    import zlib

    return int(zlib.crc32("/".join(str(x) for x in parts).encode("utf-8")) & 0x7FFFFFFF)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

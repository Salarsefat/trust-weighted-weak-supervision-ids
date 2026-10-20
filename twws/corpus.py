"""Corpus assembly: stream of all retained flows, sampled evaluation corpus and partition protocols."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .extract import connection_ids, deduplicate, load_flows
from .features import availability_time, build_base_features
from .labels import AMBIGUOUS, BENIGN, FAMILY_RULE, estimate_utc_offset, label_flows, label_summary, load_schedule
from .utils import ensure_dir, get_logger, read_json, release_memory, write_json

log = get_logger("twws.corpus")

TRAIN, VAL_A, VAL_B, TEST, UNUSED = 0, 1, 2, 3, -1
SPLIT_NAMES = {TRAIN: "train", VAL_A: "val_a", VAL_B: "val_b", TEST: "test", UNUSED: "unused"}
META_COLUMNS = ["conn", "dir", "ts", "t_avail", "day", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
                "pkts", "bytes", "rev_pkts", "rev_bytes", "duration", "conn_duration", "pay", "cap", "entropy",
                "syn", "fin", "rst", "psh", "ack", "urg", "tls", "tls_self_signed", "snort_alert", "snort_alert_any",
                "family", "attack", "label"]


# --------------------------------------------------------------------------- stream
def assemble_stream(flows: pd.DataFrame, schedule_path: str, snort_conn: pd.DataFrame | None = None,
                    dedup_tol_s: float = 1.0, deduplicated: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """De-duplicate, attach connection ids, schedule-based labels and the Snort flag.

    ``deduplicated``: the statistics of ``deduplicate`` when the caller has already de-duplicated ``flows``.
    """
    info: dict = {"flows_extracted": int(len(flows))}
    if deduplicated is None:
        flows, dstats = deduplicate(flows, dedup_tol_s)
    else:                       # de-duplicated by the caller, which then no longer holds the full table
        info["flows_extracted"], dstats = int(deduplicated["flows_before"]), deduplicated
    info["deduplication"] = dstats
    flows["conn"] = connection_ids(flows)
    flows["t_avail"] = availability_time(flows)
    sched = load_schedule(schedule_path)
    off, diag = estimate_utc_offset(flows, sched)
    lab = label_flows(flows, sched, off)
    # the string arrays themselves: .to_numpy() would create one Python object per row
    flows["family"] = lab["family"].astype(str).array
    flows["attack"] = lab["attack"].astype(str).array
    flows["label"] = lab["label"].to_numpy()
    info["utc_offset_hours"] = int(off)
    info["family_rule"] = FAMILY_RULE
    info["utc_offset_diagnostic"] = diag.to_dict("records")
    info["label_summary"] = label_summary(flows, lab).to_dict("records")
    if "snort_alert" not in flows.columns:
        flows["snort_alert"] = False
    if snort_conn is not None and len(snort_conn):
        hit = set(snort_conn.loc[snort_conn["snort_alert"], "conn"].tolist())
        flows["snort_alert"] = flows["conn"].isin(hit).to_numpy()
    flows["snort_alert"] = flows["snort_alert"].astype(bool)
    info["flows_stream"] = int(len(flows))
    info["connections_stream"] = int(flows["conn"].nunique())
    info["snort_alert_flows"] = int(flows["snort_alert"].sum())
    return flows.reset_index(drop=True), info


def connection_table(flows: pd.DataFrame) -> pd.DataFrame:
    """One row per connection with originator/responder endpoints (input of the Snort mapping)."""
    first = flows[["conn", "dir", "proto", "orig_ip", "resp_ip", "src_port", "dst_port", "ts",
                   "conn_duration"]].drop_duplicates("conn")
    is_o = first["dir"].to_numpy() == 0
    return pd.DataFrame({
        "conn": first["conn"].to_numpy(), "proto": first["proto"].astype(str).array,
        "orig_ip": first["orig_ip"].astype(str).array, "resp_ip": first["resp_ip"].astype(str).array,
        "orig_port": np.where(is_o, first["src_port"].to_numpy(), first["dst_port"].to_numpy()),
        "resp_port": np.where(is_o, first["dst_port"].to_numpy(), first["src_port"].to_numpy()),
        "ts": first["ts"].to_numpy(), "conn_duration": first["conn_duration"].to_numpy(),
    })


# --------------------------------------------------------------------------- sampling
def sample_corpus(stream: pd.DataFrame, family_cap: int | None, benign_share: float | None, max_rows: int | None,
                  seed: int = 42) -> tuple[np.ndarray, dict]:
    """Seeded selection of the evaluation corpus from the stream.

    1. ``Ambiguous`` flows are never selected.
    2. Each attack family keeps all its flows if it has at most ``family_cap``; otherwise
       ``family_cap`` flows are drawn uniformly at random without replacement.
    3. Benign flows are drawn uniformly at random so that their share of the corpus equals
       ``benign_share`` (all benign flows are kept if there are too few).
    4. If the result exceeds ``max_rows`` every class is thinned by the same factor.

    Returns a boolean mask over stream rows and a report with the counts before and after.
    """
    rng = np.random.default_rng(seed)
    fam = stream["family"].astype(str).to_numpy()
    keep = np.zeros(len(stream), dtype=bool)
    report = {"seed": seed, "family_cap": family_cap, "benign_share_target": benign_share, "max_rows": max_rows,
              "available": {}, "selected": {}}
    n_mal = 0
    for f in sorted(set(fam) - {BENIGN, AMBIGUOUS}):
        idx = np.flatnonzero(fam == f)
        report["available"][f] = int(len(idx))
        if family_cap and len(idx) > family_cap:
            idx = rng.choice(idx, size=int(family_cap), replace=False)
        keep[idx] = True
        n_mal += len(idx)
    bidx = np.flatnonzero(fam == BENIGN)
    report["available"][BENIGN] = int(len(bidx))
    report["available"][AMBIGUOUS] = int((fam == AMBIGUOUS).sum())
    if benign_share and 0 < benign_share < 1 and n_mal > 0:
        want = int(round(n_mal * benign_share / (1.0 - benign_share)))
        if want < len(bidx):
            bidx = rng.choice(bidx, size=want, replace=False)
        else:
            report["note"] = "fewer benign flows available than the target share requires; all benign flows kept"
    keep[bidx] = True
    if max_rows and keep.sum() > max_rows:
        rate = max_rows / keep.sum()
        for f in sorted(set(fam[keep])):
            idx = np.flatnonzero(keep & (fam == f))
            drop = rng.choice(idx, size=len(idx) - int(round(len(idx) * rate)), replace=False)
            keep[drop] = False
        report["thinning_rate"] = float(rate)
    for f in sorted(set(fam[keep])):
        report["selected"][f] = int((keep & (fam == f)).sum())
    report["n_selected"] = int(keep.sum())
    report["malicious_share"] = float((stream["label"].to_numpy()[keep] == 1).mean()) if keep.any() else 0.0
    return keep, report


# --------------------------------------------------------------------------- partitions
def _assign(ids: np.ndarray, rng: np.random.Generator, fractions=(0.60, 0.16, 0.04, 0.20), shuffle: bool = True) -> dict:
    """Map each id to a split code following ``fractions`` (train, val_a, val_b, test)."""
    ids = np.asarray(ids)
    order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
    n = len(ids)
    b1 = int(round(fractions[0] * n))
    b2 = b1 + int(round(fractions[1] * n))
    b3 = b2 + int(round(fractions[2] * n))
    code = np.empty(n, dtype=np.int8)
    code[order[:b1]] = TRAIN
    code[order[b1:b2]] = VAL_A
    code[order[b2:b3]] = VAL_B
    code[order[b3:]] = TEST
    return dict(zip(ids.tolist(), code.tolist()))


def make_split(meta: pd.DataFrame, protocol: str = "random", seed: int = 42, forward: dict | None = None) -> np.ndarray:
    """Partition the corpus rows.

    random         class-proportional 60/20/20 by flow, uniform at random within each class
                   (benign and each attack family) - the protocol of the first submission
    conn           the same, but whole connections are assigned, so the two directions of a
                   connection always share a subset
    chrono         within each class and capture day, connections are ordered by start time:
                   the first 60% go to training, the next 20% to validation, the last 20% to test
    forward        whole capture days are assigned to training, validation and test
                   (``forward = {"train": [...], "val": [...], "test": [...]}``)
    loao:<family>  ``random``, then the family is removed from training and validation

    The validation part is split 80/20 into ``val_a`` (model selection, early stopping) and
    ``val_b`` (threshold calibration).
    """
    rng = np.random.default_rng(seed)
    n = len(meta)
    fam = meta["family"].astype(str).to_numpy()
    split = np.full(n, UNUSED, dtype=np.int8)
    base = protocol.split(":")[0]
    if base in ("random", "loao"):
        for f in sorted(set(fam)):
            idx = np.flatnonzero(fam == f)
            m = _assign(np.arange(len(idx)), rng)
            split[idx] = np.fromiter((m[i] for i in range(len(idx))), dtype=np.int8, count=len(idx))
        if base == "loao":
            held = protocol.split(":", 1)[1]
            split[(fam == held) & (split != TEST)] = UNUSED
    elif base == "conn":
        conn = meta["conn"].to_numpy()
        for f in sorted(set(fam)):
            idx = np.flatnonzero(fam == f)
            u = np.unique(conn[idx])
            m = _assign(u, rng)
            split[idx] = pd.Series(conn[idx]).map(m).to_numpy(dtype=np.int8)
    elif base == "chrono":
        conn = meta["conn"].to_numpy()
        ts = meta["ts"].to_numpy(dtype=float)
        day = meta["day"].astype(str).to_numpy()
        for f in sorted(set(fam)):
            for d in sorted(set(day[fam == f])):
                idx = np.flatnonzero((fam == f) & (day == d))
                first = pd.Series(ts[idx]).groupby(conn[idx]).min().sort_values(kind="stable")
                m = _assign(first.index.to_numpy(), rng, shuffle=False)
                split[idx] = pd.Series(conn[idx]).map(m).to_numpy(dtype=np.int8)
    elif base == "forward":
        if not forward:
            raise ValueError("forward protocol needs the day assignment")
        day = meta["day"].astype(str).to_numpy()
        conn = meta["conn"].to_numpy()
        split[np.isin(day, forward["train"])] = TRAIN
        split[np.isin(day, forward["test"])] = TEST
        vidx = np.flatnonzero(np.isin(day, forward["val"]))
        u = np.unique(conn[vidx])
        m = _assign(u, rng, fractions=(0.0, 0.8, 0.2, 0.0))
        split[vidx] = pd.Series(conn[vidx]).map(m).to_numpy(dtype=np.int8)
    else:
        raise ValueError(f"unknown protocol '{protocol}'")
    return split


def split_table(meta: pd.DataFrame, split: np.ndarray) -> list[dict]:
    """Per split: number of flows, malicious share and per-family counts."""
    fam = meta["family"].astype(str).to_numpy()
    lab = meta["label"].to_numpy()
    groups = [("train", split == TRAIN), ("validation", (split == VAL_A) | (split == VAL_B)), ("test", split == TEST)]
    rows = []
    for name, m in groups:
        counts = pd.Series(fam[m]).value_counts().sort_index().to_dict()
        rows.append({"split": name, "flows": int(m.sum()),
                     "malicious_share": float((lab[m] == 1).mean()) if m.any() else float("nan"),
                     "families": {k: int(v) for k, v in counts.items()}})
    return rows


# --------------------------------------------------------------------------- workspace I/O
class Workspace:
    """Paths and cached artefacts of one corpus (stream, sampled corpus, base features)."""

    def __init__(self, root: str | os.PathLike, capture_root: str | os.PathLike | None = None,
                 corpus_root: str | os.PathLike | None = None) -> None:
        self.root = ensure_dir(root)
        # captures, Zeek and Snort output (data/, extract/, snort/); another workspace's when they are shared
        self.capture_root = ensure_dir(capture_root) if capture_root else self.root
        # stream, corpus and pair statistics: another workspace's when only the oracles or the aggregation differ
        self.corpus_dir = ensure_dir(Path(corpus_root) / "corpus") if corpus_root else ensure_dir(self.root / "corpus")
        self.results_dir = ensure_dir(self.root / "results")
        self.cache_dir = ensure_dir(self.root / "cache")
        self._meta = self._X = self._aux = None

    # stream ---------------------------------------------------------------
    @property
    def stream_path(self) -> Path:
        return self.corpus_dir / "stream.parquet"

    def save_stream(self, stream: pd.DataFrame, info: dict, chunk_rows: int = 2_000_000) -> None:
        """Write the stream (categories as strings) in slices, so that no full copy of it is made."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp = self.stream_path.with_name(self.stream_path.name + ".partial")
        writer = None
        try:
            for a in range(0, max(len(stream), 1), chunk_rows):
                df = stream.iloc[a:a + chunk_rows]
                cat = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
                if cat:
                    df = df.assign(**{c: df[c].astype(str) for c in cat})
                if writer is None:
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    writer = pq.ParquetWriter(tmp, table.schema)
                else:
                    table = pa.Table.from_pandas(df, schema=writer.schema, preserve_index=False)
                writer.write_table(table)
                del df, table
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp, self.stream_path)
        write_json(self.corpus_dir / "stream_info.json", info)

    def load_stream(self) -> pd.DataFrame:
        """The stream with its string columns as categories (sorted), read without materializing the strings."""
        import pyarrow.parquet as pq

        cols = ["src_ip", "dst_ip", "orig_ip", "resp_ip", "proto", "conn_state", "day", "capture", "family", "attack"]
        names = pq.read_schema(self.stream_path).names
        table = pq.read_table(self.stream_path, read_dictionary=[c for c in cols if c in names])
        df = table.to_pandas(self_destruct=True, split_blocks=True)
        del table
        for c in cols:
            if c not in df.columns:
                continue
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].cat.reorder_categories(df[c].cat.categories.sort_values())
            else:
                df[c] = df[c].astype("category")
        release_memory()
        return df

    # corpus ---------------------------------------------------------------
    def build_corpus(self, stream: pd.DataFrame, cfg) -> dict:
        """Sample the corpus, compute base features for its rows and store everything."""
        if "snort_alert_any" not in stream.columns:     # streams without the attack-class distinction
            stream["snort_alert_any"] = stream["snort_alert"] if "snort_alert" in stream.columns else False
        keep, report = sample_corpus(stream, cfg.corpus.get("family_cap"), cfg.corpus.get("benign_share"),
                                     cfg.corpus.get("max_rows"), int(cfg.corpus.get("sample_seed", 42)))
        X, aux = build_base_features(stream, keep)
        meta = stream.loc[keep, [c for c in META_COLUMNS if c in stream.columns]].reset_index(drop=True)
        meta.insert(0, "stream_row", np.flatnonzero(keep))
        for c in meta.columns:
            if isinstance(meta[c].dtype, pd.CategoricalDtype):
                meta[c] = meta[c].astype(str)
        meta.to_parquet(self.corpus_dir / "meta.parquet", index=False)
        X.to_parquet(self.corpus_dir / "X_base.parquet", index=False)
        aux.to_parquet(self.corpus_dir / "aux.parquet", index=False)
        report["base_features"] = list(X.columns)
        write_json(self.corpus_dir / "corpus_info.json", report)
        self._meta, self._X, self._aux = meta, X, None
        # pair statistics of the oracles 'rate' and 'beacon' (computed over the full stream)
        self.build_pairs(stream)
        return report

    # pair statistics (oracles 'rate' and 'beacon', docs/ORACLES_EXTENDED.md) ---------------------
    @property
    def pairs_path(self) -> Path:
        return self.corpus_dir / "aux_pairs.parquet"

    def build_pairs(self, stream: pd.DataFrame | None = None) -> dict:
        """Compute the pair statistics of the corpus rows over the full stream and store them."""
        import time

        from .pairs import STREAM_COLUMNS, pair_features

        t0 = time.perf_counter()
        if stream is None:
            if not self.stream_path.exists():
                raise FileNotFoundError(f"{self.stream_path} does not exist: build the corpus first")
            stream = pd.read_parquet(self.stream_path, columns=STREAM_COLUMNS)
        rows = self.meta["stream_row"].to_numpy()
        pf = pair_features(stream, rows)
        tmp = self.pairs_path.with_name(self.pairs_path.name + ".partial")
        pf.to_parquet(tmp, index=False)
        os.replace(tmp, self.pairs_path)
        self._aux = None
        info = {"rows": int(len(pf)), "tcp_rows": int((pf["aux_pair_id"] >= 0).sum()),
                "pairs_in_corpus": int(pf.loc[pf["aux_pair_id"] >= 0, "aux_pair_id"].nunique()),
                "stream_records": int(len(stream)), "seconds": round(time.perf_counter() - t0, 1)}
        write_json(self.corpus_dir / "pairs_info.json", info)
        log.info("pair statistics stored: %s", info)
        return info

    @property
    def meta(self) -> pd.DataFrame:
        if self._meta is None:
            self._meta = pd.read_parquet(self.corpus_dir / "meta.parquet")
        return self._meta

    @property
    def X_base(self) -> pd.DataFrame:
        if self._X is None:
            self._X = pd.read_parquet(self.corpus_dir / "X_base.parquet")
        return self._X

    @property
    def aux(self) -> pd.DataFrame:
        """Oracle inputs: aux.parquet, plus the pair statistics (aux_pairs.parquet) when they exist."""
        if self._aux is None:
            aux = pd.read_parquet(self.corpus_dir / "aux.parquet")
            if self.pairs_path.exists():
                pairs = pd.read_parquet(self.pairs_path)
                if len(pairs) != len(aux):
                    raise RuntimeError(f"{self.pairs_path} has {len(pairs)} rows, the corpus {len(aux)}: run "
                                       "`twws.cli pairs` again")
                aux = pd.concat([aux.reset_index(drop=True), pairs.reset_index(drop=True)], axis=1)
            self._aux = aux
        return self._aux

    @property
    def has_pairs(self) -> bool:
        return self.pairs_path.exists()

    def info(self, name: str) -> dict:
        p = self.corpus_dir / f"{name}.json"
        return read_json(p) if p.exists() else {}

    def result_path(self, name: str) -> Path:
        return self.results_dir / f"{name}.json"

    def has_result(self, name: str) -> bool:
        return self.result_path(name).exists()

    def save_result(self, name: str, obj: dict) -> None:
        write_json(self.result_path(name), obj)
        log.info("result written: %s", self.result_path(name))

    def load_result(self, name: str) -> dict:
        return read_json(self.result_path(name))


def load_flows_for_corpus(extract_dir: str | os.PathLike, days: list[str] | None = None) -> pd.DataFrame:
    return load_flows(extract_dir, days)

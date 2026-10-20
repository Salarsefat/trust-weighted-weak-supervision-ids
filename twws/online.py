"""Step 10: streaming feature engine, online normalisation and the inference loop.

``OnlineFeatureEngine`` keeps, per source host, the records of the last hour and running
window statistics.  It implements exactly the definition of ``twws.features`` (aggregates
over records that became available strictly before the current one) and is tested against the
offline kernel.  ``OnlineDetector`` chains the engine, imputation/scaling, the autoencoder,
feature selection and the boosted ensemble, and times each stage.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from .features import HOST_FEATURES, N_AGG, STAT_FEATURES, WINDOWS, statistical_features, temporal_feature_names

NAN = float("nan")


class _Host:
    __slots__ = ("t", "rec", "hi", "lo", "R", "nd", "npt", "S", "cd", "cp", "base")

    def __init__(self, nw: int) -> None:
        self.t: list = []            # availability times of buffered records
        self.rec: list = []          # (rb, rp, lb, lp, ld, is_o, norep, syn, dst, dport)
        self.hi = 0                  # records [lo[w], hi) are inside window w
        self.lo = [0] * nw
        self.R = [[0.0] * 11 for _ in range(nw)]
        self.nd = [0] * nw
        self.npt = [0] * nw
        self.S = [0.0] * nw
        self.cd = [defaultdict(int) for _ in range(nw)]
        self.cp = [defaultdict(int) for _ in range(nw)]
        self.base = 0                # number of records already pruned from the front


class OnlineFeatureEngine:
    """Per-record computation of the host-context and temporal features."""

    def __init__(self, windows: dict[str, float] | None = None, prune_every: int = 4096) -> None:
        self.windows = dict(windows or WINDOWS)
        self.wins = list(self.windows.values())
        self.nw = len(self.wins)
        self.w1m = self.wins.index(60.0) if 60.0 in self.wins else -1
        self.hosts: dict = {}
        self.prune_every = prune_every
        self.n_seen = 0
        self.names = HOST_FEATURES + temporal_feature_names(self.windows)

    def state_records(self) -> int:
        return sum(len(h.t) for h in self.hosts.values())

    def update(self, host, t: float, rb: float, rp: float, dur: float, is_o: float, norep: float, syn: float,
               dst, dport: int, out: np.ndarray) -> None:
        """Write the 3 host-context and ``nw * N_AGG`` temporal features of this record into ``out``."""
        h = self.hosts.get(host)
        if h is None:
            h = self.hosts[host] = _Host(self.nw)
        ts, recs, wins, nw = h.t, h.rec, self.wins, self.nw
        n_buf = len(ts)
        # 1. records that became available strictly earlier enter every window
        hi = h.hi
        while hi < n_buf and ts[hi] < t:
            r = recs[hi]
            for w in range(nw):
                cd, cp, R = h.cd[w], h.cp[w], h.R[w]
                c = cd[r[8]]
                if c == 0:
                    h.nd[w] += 1
                cd[r[8]] = c + 1
                c = cp[r[9]]
                if c == 0:
                    h.npt[w] += 1
                else:
                    h.S[w] -= c * math.log(c)
                c += 1
                h.S[w] += c * math.log(c)
                cp[r[9]] = c
                R[0] += r[0]
                R[1] += r[1]
                R[2] += r[2]
                R[3] += r[2] * r[2]
                R[4] += r[3]
                R[5] += r[3] * r[3]
                R[6] += r[4]
                R[7] += r[4] * r[4]
                R[8] += r[5]
                R[9] += r[6]
                R[10] += r[7]
            hi += 1
        h.hi = hi
        # 2. expire records older than each window
        for w in range(nw):
            lim = t - wins[w]
            lo = h.lo[w]
            cd, cp, R = h.cd[w], h.cp[w], h.R[w]
            while lo < hi and ts[lo] < lim:
                r = recs[lo]
                c = cd[r[8]] - 1
                if c == 0:
                    del cd[r[8]]
                    h.nd[w] -= 1
                else:
                    cd[r[8]] = c
                c = cp[r[9]]
                h.S[w] -= c * math.log(c)
                c -= 1
                if c == 0:
                    del cp[r[9]]
                    h.npt[w] -= 1
                else:
                    h.S[w] += c * math.log(c)
                    cp[r[9]] = c
                R[0] -= r[0]
                R[1] -= r[1]
                R[2] -= r[2]
                R[3] -= r[2] * r[2]
                R[4] -= r[3]
                R[5] -= r[3] * r[3]
                R[6] -= r[4]
                R[7] -= r[4] * r[4]
                R[8] -= r[5]
                R[9] -= r[6]
                R[10] -= r[7]
                lo += 1
            h.lo[w] = lo
            if lo == hi:
                h.S[w] = 0.0
                for q in range(11):
                    R[q] = 0.0
        # 3. features
        for w in range(nw):
            n = hi - h.lo[w]
            b = 3 + w * N_AGG
            R = h.R[w]
            out[b] = n
            if n > 0:
                out[b + 1] = math.log1p(max(R[0], 0.0))
                out[b + 2] = math.log1p(max(R[1], 0.0))
                mu = R[2] / n
                out[b + 3] = mu
                out[b + 4] = math.sqrt(max(R[3] / n - mu * mu, 0.0)) if n > 1 else 0.0
                mu = R[4] / n
                out[b + 5] = mu
                out[b + 6] = math.sqrt(max(R[5] / n - mu * mu, 0.0)) if n > 1 else 0.0
                mu = R[6] / n
                out[b + 7] = mu
                out[b + 8] = math.sqrt(max(R[7] / n - mu * mu, 0.0)) if n > 1 else 0.0
                out[b + 9] = h.nd[w]
                out[b + 10] = h.npt[w]
                e = math.log(n) - h.S[w] / n
                out[b + 11] = e if e > 0.0 else 0.0
                out[b + 12] = R[8] / n
                out[b + 13] = R[9] / n
            else:
                out[b + 1] = 0.0
                out[b + 2] = 0.0
                out[b + 3: b + 14] = NAN
                out[b + 9] = 0.0
                out[b + 10] = 0.0
        if self.w1m >= 0:
            n = hi - h.lo[self.w1m]
            out[0] = n / 60.0
            out[2] = h.R[self.w1m][8] / n if n > 0 else NAN
        else:
            out[0] = NAN
            out[2] = NAN
        out[1] = math.log1p(t - ts[hi - 1]) if hi > 0 else NAN
        # 4. buffer the current record (it enters the windows of later records)
        ts.append(t)
        recs.append((rb, rp, math.log1p(rb), math.log1p(rp), math.log1p(dur), is_o, norep, syn, dst, dport))
        self.n_seen += 1
        if len(ts) >= self.prune_every and min(h.lo) > len(ts) // 2:
            k = min(min(h.lo), h.hi - 1)       # keep the latest entered record (inter-arrival time)
            del ts[:k]
            del recs[:k]
            h.hi -= k
            h.lo = [x - k for x in h.lo]
            h.base += k

    def batch(self, df: pd.DataFrame) -> np.ndarray:
        """Host-context + temporal features for a micro-batch (records in availability order)."""
        n = len(df)
        out = np.empty((n, 3 + self.nw * N_AGG), dtype=np.float64)
        host = df["src_ip"].to_numpy()
        t = df["t_avail"].to_numpy(dtype=np.float64)
        rb = df["bytes"].to_numpy(dtype=np.float64)
        rp = df["pkts"].to_numpy(dtype=np.float64)
        dur = df["duration"].to_numpy(dtype=np.float64)
        is_o = (df["dir"].to_numpy() == 0).astype(np.float64)
        nr = (df["rev_pkts"].to_numpy() == 0).astype(np.float64)
        syn = df["syn"].to_numpy(dtype=np.float64)
        dst = df["dst_ip"].to_numpy()
        dport = df["dst_port"].to_numpy()
        up = self.update
        for i in range(n):
            up(host[i], t[i], rb[i], rp[i], dur[i], is_o[i], nr[i], syn[i], dst[i], int(dport[i]), out[i])
        return out


class SlidingScaler:
    """Median/IQR scaling with statistics over the most recent ``window`` values per feature.

    Statistics are refreshed once per micro-batch from a ring buffer; until the buffer holds
    ``window`` values, and whenever an IQR is zero, the training statistics are used.
    """

    def __init__(self, median: np.ndarray, iqr: np.ndarray, window: int = 10_000) -> None:
        self.med0, self.iqr0 = np.asarray(median, np.float64), np.asarray(iqr, np.float64)
        self.window = int(window)
        self.buf = np.full((self.window, len(self.med0)), np.nan)
        self.pos = 0
        self.filled = 0
        self.med, self.iqr = self.med0.copy(), self.iqr0.copy()

    def transform(self, A: np.ndarray) -> np.ndarray:
        """Scale ``A`` (raw, imputed values) with the current statistics, then absorb it."""
        out = (A - self.med) / self.iqr
        n = len(A)
        if n >= self.window:
            self.buf[:] = A[-self.window:]
            self.pos, self.filled = 0, self.window
        else:
            end = self.pos + n
            if end <= self.window:
                self.buf[self.pos:end] = A
            else:
                k = self.window - self.pos
                self.buf[self.pos:] = A[:k]
                self.buf[: end - self.window] = A[k:]
            self.pos = end % self.window
            self.filled = min(self.window, self.filled + n)
        if self.filled >= self.window:
            q1, med, q3 = np.nanpercentile(self.buf, [25, 50, 75], axis=0)
            iqr = q3 - q1
            ok = np.isfinite(iqr) & (iqr > 0) & np.isfinite(med)
            self.med = np.where(ok, med, self.med0)
            self.iqr = np.where(ok, iqr, self.iqr0)
        return out


class OnlineDetector:
    """Record in, verdict out: engine -> imputation/scaling -> autoencoder -> selection -> ensemble."""

    def __init__(self, feature_pipeline, selector, model, threshold: float, base_columns: list[str],
                 sliding_window: int | None = None) -> None:
        self.fp, self.selector, self.model, self.threshold = feature_pipeline, selector, model, float(threshold)
        self.engine = OnlineFeatureEngine()
        self.base_columns = list(base_columns)
        self.host_cols = HOST_FEATURES + temporal_feature_names()
        self.use_host = [c for c in self.host_cols if c in self.base_columns]
        self.host_pos = [self.host_cols.index(c) for c in self.use_host]
        self.stat_cols = [c for c in self.base_columns if c in STAT_FEATURES]
        self.sliding = None
        if sliding_window:
            self.sliding = SlidingScaler(self.fp.pre.median_, self.fp.pre.iqr_, sliding_window)
        self.timers = {"features": 0.0, "normalise": 0.0, "model": 0.0}
        self.batches = 0

    def process(self, batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, verdicts) for one micro-batch of finalised flow records."""
        t0 = time.perf_counter()
        H = self.engine.batch(batch)
        S = statistical_features(batch)
        Xb = pd.DataFrame(np.column_stack([S[self.stat_cols].to_numpy(dtype=np.float64), H[:, self.host_pos]]),
                          columns=self.stat_cols + self.use_host)[self.base_columns]
        t1 = time.perf_counter()
        if self.sliding is None:
            X = self.fp.transform(Xb)
        else:
            X = self._transform_sliding(Xb)
        Xs = self.selector.transform(X)
        t2 = time.perf_counter()
        score = self.model.predict_proba(np.ascontiguousarray(Xs, dtype=np.float32))
        t3 = time.perf_counter()
        self.timers["features"] += t1 - t0
        self.timers["normalise"] += t2 - t1
        self.timers["model"] += t3 - t2
        self.batches += 1
        return score, score > self.threshold

    def _transform_sliding(self, Xb: pd.DataFrame) -> np.ndarray:
        pre = self.fp.pre
        scale_flag = pre.scale
        pre.scale = False                      # impute only; scaling is done with sliding statistics
        try:
            A = pre.transform(Xb).astype(np.float64)
        finally:
            pre.scale = scale_flag
        A = self.sliding.transform(A).astype(np.float32)
        out = np.empty((len(A), len(self.fp.columns)), dtype=np.float32)
        nb = A.shape[1]
        out[:, :nb] = A
        if self.fp.use_latent:
            Z = self.fp.ae.transform(A[:, self.fp.stat_idx_])
            out[:, nb:] = (Z - self.fp.z_median_) / self.fp.z_iqr_
        return out

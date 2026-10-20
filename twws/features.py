"""Feature construction (Step 3): statistical, causal temporal and autoencoder latent features.

Causality.  A flow can be classified only when its record is complete, i.e. when the
connection ends.  Host-level aggregates are therefore indexed by the *availability time* of a
record, ``t_avail = connection start + connection duration``, and an aggregate for flow *i*
uses only records of the same source host with ``t_avail`` strictly smaller than that of
flow *i* and not older than the window length.  The offline kernel below (Numba) and the
streaming engine in ``twws.online`` implement the same definition and are tested for equality.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("twws.features")

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a declared dependency
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


WINDOWS: dict[str, float] = {"5s": 5.0, "1m": 60.0, "5m": 300.0, "1h": 3600.0}
TEMPORAL_AGGS = ["cnt", "bytes_sum", "pkts_sum", "lbytes_mean", "lbytes_std", "lpkts_mean", "lpkts_std",
                 "ldur_mean", "ldur_std", "n_dst", "n_dport", "dport_entropy", "frac_orig", "frac_noreply"]
N_AGG = len(TEMPORAL_AGGS)

STAT_FEATURES = [
    "s_duration", "s_pkts", "s_bytes", "s_bytes_per_pkt", "s_pay_per_pkt", "s_pkts_per_s", "s_bytes_per_s",
    "s_rev_pkts", "s_rev_bytes", "s_pkt_ratio", "s_byte_ratio",
    "s_syn_frac", "s_fin_frac", "s_rst_frac", "s_psh_frac", "s_ack_frac", "s_urg_frac",
    "s_entropy", "s_entropy_missing", "s_payload_full", "s_tls", "s_proto_tcp", "s_proto_udp", "s_is_orig",
    "s_dport_wellknown", "s_sport_wellknown", "s_no_reply",
]
HOST_FEATURES = ["h_conn_rate_1m", "h_iat", "h_outbound_ratio_1m"]   # single-window host context (f2, f4, f3)
AUX_COLUMNS = ["aux_syn_ratio_1m", "aux_bpp_1m", "aux_fanout_5m"]    # used by oracles only, never by the model
NEVER_IMPUTE = ["s_entropy"]


def temporal_feature_names(windows: dict[str, float] | None = None) -> list[str]:
    windows = windows or WINDOWS
    return [f"t_{w}_{a}" for w in windows for a in TEMPORAL_AGGS]


def availability_time(flows: pd.DataFrame) -> np.ndarray:
    return flows["ts"].to_numpy(dtype=np.float64) + flows["conn_duration"].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- statistical features
def statistical_features(flows: pd.DataFrame) -> pd.DataFrame:
    """Per-flow statistical features (no host context); float32, NaN where undefined."""
    pk = flows["pkts"].to_numpy(dtype=np.float64)
    by = flows["bytes"].to_numpy(dtype=np.float64)
    rp = flows["rev_pkts"].to_numpy(dtype=np.float64)
    rb = flows["rev_bytes"].to_numpy(dtype=np.float64)
    dur = flows["duration"].to_numpy(dtype=np.float64)
    pay = flows["pay"].to_numpy(dtype=np.float64)
    cap = flows["cap"].to_numpy(dtype=np.float64)
    proto = flows["proto"].astype(str).to_numpy()
    pks = np.maximum(pk, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pps = np.where(dur > 0, pk / dur, np.nan)
        bps = np.where(dur > 0, by / dur, np.nan)
        pkt_ratio = pk / (pk + rp)
        byte_ratio = np.where(by + rb > 0, by / (by + rb), np.nan)
    ent = flows["entropy"].to_numpy(dtype=np.float64)
    out = {
        "s_duration": np.log1p(dur), "s_pkts": np.log1p(pk), "s_bytes": np.log1p(by),
        "s_bytes_per_pkt": by / pks, "s_pay_per_pkt": pay / pks,
        "s_pkts_per_s": np.log1p(pps), "s_bytes_per_s": np.log1p(bps),
        "s_rev_pkts": np.log1p(rp), "s_rev_bytes": np.log1p(rb),
        "s_pkt_ratio": pkt_ratio, "s_byte_ratio": byte_ratio,
    }
    for fl in ["syn", "fin", "rst", "psh", "ack", "urg"]:
        out[f"s_{fl}_frac"] = np.minimum(flows[fl].to_numpy(dtype=np.float64) / pks, 1.0)
    out["s_entropy"] = ent
    out["s_entropy_missing"] = np.isnan(ent).astype(np.float64)
    out["s_payload_full"] = (cap >= pay).astype(np.float64)
    out["s_tls"] = flows["tls"].to_numpy(dtype=np.float64)
    out["s_proto_tcp"] = (proto == "tcp").astype(np.float64)
    out["s_proto_udp"] = (proto == "udp").astype(np.float64)
    out["s_is_orig"] = (flows["dir"].to_numpy() == 0).astype(np.float64)
    out["s_dport_wellknown"] = (flows["dst_port"].to_numpy() < 1024).astype(np.float64)
    out["s_sport_wellknown"] = (flows["src_port"].to_numpy() < 1024).astype(np.float64)
    out["s_no_reply"] = (rp == 0).astype(np.float64)
    return pd.DataFrame({k: out[k].astype(np.float32) for k in STAT_FEATURES})


# --------------------------------------------------------------------------- temporal kernel
@njit(cache=True)
def _temporal_kernel(seg, t, lb, lp, ld, rb, rpk, is_o, norep, syn, dst, dport, wins, need, out_idx,
                     out_t, out_h, out_aux, n_dst_codes):
    """One pass over records sorted by (source host, availability time).

    seg       start offsets of the host segments (len = hosts + 1)
    wins      window lengths in seconds
    need      1 where the record's features must be written
    out_idx   row of the output arrays for a needed record
    out_t     (rows, len(wins) * N_AGG) temporal aggregates over strictly earlier records
    out_h     (rows, 3) connection rate (1 m), inter-arrival time, outbound ratio (1 m)
    out_aux   (rows, 3) SYN ratio and bytes/packet over the last minute incl. the current
              record; the third column is filled by ``_fanout_kernel``

    Window sums are running sums (add when a record enters, subtract when it expires) that
    are reset to exactly zero whenever a window empties; the streaming engine performs the
    same operations in the same order.
    """
    nw = len(wins)
    nagg = 14
    cnt_dst = np.zeros((nw, n_dst_codes), dtype=np.int32)
    cnt_port = np.zeros((nw, 65536), dtype=np.int32)
    w1m = -1
    w5m = -1
    for w in range(nw):
        if wins[w] == 60.0:
            w1m = w
        if wins[w] == 300.0:
            w5m = w
    lo = np.zeros(nw, dtype=np.int64)
    nd = np.zeros(nw, dtype=np.int64)
    npt = np.zeros(nw, dtype=np.int64)
    S = np.zeros(nw)
    # running sums per window: rb, rp, lb, lb2, lp, lp2, ld, ld2, orig, noreply, syn
    R = np.zeros((nw, 11))
    for s in range(len(seg) - 1):
        a = seg[s]
        b = seg[s + 1]
        hi = a                                  # window of w = [lo[w], hi) in global indices
        for w in range(nw):
            lo[w] = a
            nd[w] = 0
            npt[w] = 0
            S[w] = 0.0
            for q in range(11):
                R[w, q] = 0.0
        for k in range(a, b):
            ti = t[k]
            # records that became available strictly before this one enter every window
            while hi < k and t[hi] < ti:
                for w in range(nw):
                    c = cnt_dst[w, dst[hi]]
                    if c == 0:
                        nd[w] += 1
                    cnt_dst[w, dst[hi]] = c + 1
                    c = cnt_port[w, dport[hi]]
                    if c == 0:
                        npt[w] += 1
                    else:
                        S[w] -= c * math.log(c)
                    c += 1
                    S[w] += c * math.log(c)
                    cnt_port[w, dport[hi]] = c
                    R[w, 0] += rb[hi]
                    R[w, 1] += rpk[hi]
                    R[w, 2] += lb[hi]
                    R[w, 3] += lb[hi] * lb[hi]
                    R[w, 4] += lp[hi]
                    R[w, 5] += lp[hi] * lp[hi]
                    R[w, 6] += ld[hi]
                    R[w, 7] += ld[hi] * ld[hi]
                    R[w, 8] += is_o[hi]
                    R[w, 9] += norep[hi]
                    R[w, 10] += syn[hi]
                hi += 1
            for w in range(nw):
                lim = ti - wins[w]
                while lo[w] < hi and t[lo[w]] < lim:
                    kk = lo[w]
                    c = cnt_dst[w, dst[kk]] - 1
                    cnt_dst[w, dst[kk]] = c
                    if c == 0:
                        nd[w] -= 1
                    c = cnt_port[w, dport[kk]]
                    S[w] -= c * math.log(c)
                    c -= 1
                    if c == 0:
                        npt[w] -= 1
                    else:
                        S[w] += c * math.log(c)
                    cnt_port[w, dport[kk]] = c
                    R[w, 0] -= rb[kk]
                    R[w, 1] -= rpk[kk]
                    R[w, 2] -= lb[kk]
                    R[w, 3] -= lb[kk] * lb[kk]
                    R[w, 4] -= lp[kk]
                    R[w, 5] -= lp[kk] * lp[kk]
                    R[w, 6] -= ld[kk]
                    R[w, 7] -= ld[kk] * ld[kk]
                    R[w, 8] -= is_o[kk]
                    R[w, 9] -= norep[kk]
                    R[w, 10] -= syn[kk]
                    lo[w] += 1
                if lo[w] == hi:                 # empty window: remove accumulated rounding error
                    S[w] = 0.0
                    for q in range(11):
                        R[w, q] = 0.0
            if need[k] == 1:
                r = out_idx[k]
                for w in range(nw):
                    n = hi - lo[w]
                    base = w * nagg
                    out_t[r, base + 0] = n
                    if n > 0:
                        out_t[r, base + 1] = math.log1p(max(R[w, 0], 0.0))
                        out_t[r, base + 2] = math.log1p(max(R[w, 1], 0.0))
                        mu = R[w, 2] / n
                        out_t[r, base + 3] = mu
                        out_t[r, base + 4] = math.sqrt(max(R[w, 3] / n - mu * mu, 0.0)) if n > 1 else 0.0
                        mu = R[w, 4] / n
                        out_t[r, base + 5] = mu
                        out_t[r, base + 6] = math.sqrt(max(R[w, 5] / n - mu * mu, 0.0)) if n > 1 else 0.0
                        mu = R[w, 6] / n
                        out_t[r, base + 7] = mu
                        out_t[r, base + 8] = math.sqrt(max(R[w, 7] / n - mu * mu, 0.0)) if n > 1 else 0.0
                        out_t[r, base + 9] = nd[w]
                        out_t[r, base + 10] = npt[w]
                        h = math.log(n) - S[w] / n
                        out_t[r, base + 11] = h if h > 0.0 else 0.0
                        out_t[r, base + 12] = R[w, 8] / n
                        out_t[r, base + 13] = R[w, 9] / n
                    else:
                        out_t[r, base + 1] = 0.0
                        out_t[r, base + 2] = 0.0
                        for q in range(3, 14):
                            out_t[r, base + q] = np.nan
                        out_t[r, base + 9] = 0.0
                        out_t[r, base + 10] = 0.0
                # single-window host context (strictly earlier records)
                if w1m >= 0:
                    n = hi - lo[w1m]
                    out_h[r, 0] = n / 60.0
                    out_h[r, 2] = R[w1m, 8] / n if n > 0 else np.nan
                    # oracle inputs: last minute including the current record
                    sp = R[w1m, 1] + rpk[k]
                    out_aux[r, 0] = (R[w1m, 10] + syn[k]) / sp if sp > 0 else np.nan
                    out_aux[r, 1] = (R[w1m, 0] + rb[k]) / sp if sp > 0 else np.nan
                else:
                    out_h[r, 0] = np.nan
                    out_h[r, 2] = np.nan
                    out_aux[r, 0] = np.nan
                    out_aux[r, 1] = np.nan
                out_h[r, 1] = math.log1p(ti - t[hi - 1]) if hi > a else np.nan
                if w5m >= 0:
                    out_aux[r, 2] = nd[w5m] + (1 if cnt_dst[w5m, dst[k]] == 0 else 0)
                else:
                    out_aux[r, 2] = np.nan
        # reset the counters touched by this host
        for w in range(nw):
            for q in range(lo[w], hi):
                cnt_dst[w, dst[q]] = 0
                cnt_port[w, dport[q]] = 0


@njit(cache=True)
def _fanout_kernel(seg, t, dst, is_o, window, need, out_idx, out, n_codes):
    """Distinct destinations contacted by connections the host *originated* during the last
    ``window`` seconds (strictly earlier records plus the current one when it is originated)."""
    cnt = np.zeros(n_codes, dtype=np.int32)
    for s in range(len(seg) - 1):
        a = seg[s]
        b = seg[s + 1]
        lo = a
        hi = a
        nd = 0
        for k in range(a, b):
            ti = t[k]
            while hi < k and t[hi] < ti:
                if is_o[hi] == 1.0:
                    c = cnt[dst[hi]]
                    if c == 0:
                        nd += 1
                    cnt[dst[hi]] = c + 1
                hi += 1
            lim = ti - window
            while lo < hi and t[lo] < lim:
                if is_o[lo] == 1.0:
                    c = cnt[dst[lo]] - 1
                    cnt[dst[lo]] = c
                    if c == 0:
                        nd -= 1
                lo += 1
            if need[k] == 1:
                extra = 1 if (is_o[k] == 1.0 and cnt[dst[k]] == 0) else 0
                out[out_idx[k]] = nd + extra
        for q in range(lo, hi):
            cnt[dst[q]] = 0


def host_stream_order(flows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort order by (source host, availability time, row) and the host segment offsets."""
    host = pd.factorize(flows["src_ip"].astype(str), sort=False)[0].astype(np.int64)
    t = availability_time(flows)
    order = np.lexsort((np.arange(len(flows)), t, host))
    hs = host[order]
    cut = np.flatnonzero(np.diff(hs)) + 1
    seg = np.concatenate(([0], cut, [len(flows)])).astype(np.int64)
    return order, seg, t, int(host.max()) + 1 if len(host) else 0


def temporal_features(flows: pd.DataFrame, need: np.ndarray | None = None,
                      windows: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Causal host-level aggregates for the rows flagged in ``need`` (all rows if ``None``).

    The aggregates of a row use every *stream* record of the same source host that became
    available earlier, whether or not that record is itself flagged in ``need``.

    Returns (temporal features, single-window host features, auxiliary oracle inputs), each
    with one row per needed record, in the original row order of the needed records.
    """
    windows = windows or WINDOWS
    n = len(flows)
    if need is None:
        need = np.ones(n, dtype=bool)
    need = np.asarray(need, dtype=bool)
    rows = np.flatnonzero(need)
    out_row = np.full(n, -1, dtype=np.int64)
    out_row[rows] = np.arange(len(rows))
    order, seg, t, _ = host_stream_order(flows)
    pk = flows["pkts"].to_numpy(dtype=np.float64)
    by = flows["bytes"].to_numpy(dtype=np.float64)
    dur = flows["duration"].to_numpy(dtype=np.float64)
    dst_code, dst_uniq = pd.factorize(flows["dst_ip"].astype(str), sort=False)
    wins = np.array(list(windows.values()), dtype=np.float64)
    out_t = np.zeros((len(rows), len(wins) * N_AGG), dtype=np.float32)
    out_h = np.zeros((len(rows), 3), dtype=np.float32)
    out_aux = np.zeros((len(rows), 3), dtype=np.float32)
    if n:
        _temporal_kernel(
            seg, t[order], np.log1p(by)[order], np.log1p(pk)[order], np.log1p(dur)[order], by[order], pk[order],
            (flows["dir"].to_numpy() == 0).astype(np.float64)[order],
            (flows["rev_pkts"].to_numpy() == 0).astype(np.float64)[order],
            flows["syn"].to_numpy(dtype=np.float64)[order],
            dst_code.astype(np.int64)[order], np.clip(flows["dst_port"].to_numpy(), 0, 65535).astype(np.int64)[order],
            wins, need.astype(np.int8)[order], out_row[order], out_t, out_h, out_aux, max(len(dst_uniq), 1))
        fan = np.zeros(len(rows), dtype=np.float32)
        _fanout_kernel(seg, t[order], dst_code.astype(np.int64)[order],
                       (flows["dir"].to_numpy() == 0).astype(np.float64)[order], 300.0,
                       need.astype(np.int8)[order], out_row[order], fan, max(len(dst_uniq), 1))
        out_aux[:, 2] = fan
    tf = pd.DataFrame(out_t, columns=temporal_feature_names(windows))
    hf = pd.DataFrame(out_h, columns=HOST_FEATURES)
    af = pd.DataFrame(out_aux, columns=AUX_COLUMNS)
    return tf, hf, af


def build_base_features(flows: pd.DataFrame, need: np.ndarray | None = None,
                        windows: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Statistical + host-context + temporal features and the oracle auxiliaries for needed rows."""
    if need is None:
        need = np.ones(len(flows), dtype=bool)
    stat = statistical_features(flows.loc[np.asarray(need, dtype=bool)]).reset_index(drop=True)
    tf, hf, af = temporal_features(flows, need, windows)
    X = pd.concat([stat, hf, tf], axis=1)
    return X, af


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Map every feature to its subspace: statistical (incl. host context), temporal, latent."""
    g = {"statistical": [], "temporal": [], "latent": []}
    for c in columns:
        if c.startswith("t_"):
            g["temporal"].append(c)
        elif c.startswith("z_"):
            g["latent"].append(c)
        else:
            g["statistical"].append(c)
    return g


# --------------------------------------------------------------------------- latent features
class LatentAE:
    """Stacked autoencoder on the scaled statistical sub-vector (trained without labels).

    Architecture  d_in -> hidden -> code -> hidden -> d_in, ReLU activations, linear output,
    mean-squared reconstruction error, Adam.  Early stopping on a held-out part of the
    training rows.  All settings are recorded in ``self.info``.
    """

    def __init__(self, code: int = 32, hidden: int = 64, lr: float = 1e-3, batch_size: int = 2048,
                 max_epochs: int = 30, patience: int = 3, holdout: float = 0.1, seed: int = 0,
                 max_rows: int | None = None, threads: int | None = None) -> None:
        self.code, self.hidden, self.lr, self.batch_size = code, hidden, lr, batch_size
        self.max_epochs, self.patience, self.holdout, self.seed = max_epochs, patience, holdout, seed
        self.max_rows, self.threads = max_rows, threads
        self.model = None
        self.info: dict = {}

    def _build(self, d_in: int):
        import torch.nn as nn

        enc = nn.Sequential(nn.Linear(d_in, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.code), nn.ReLU())
        dec = nn.Sequential(nn.Linear(self.code, self.hidden), nn.ReLU(), nn.Linear(self.hidden, d_in))
        return nn.ModuleDict({"enc": enc, "dec": dec})

    def fit(self, X: np.ndarray) -> "LatentAE":
        import torch

        if self.threads:
            torch.set_num_threads(int(self.threads))
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -20.0, 20.0)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        if self.max_rows and len(idx) > self.max_rows:
            idx = idx[: self.max_rows]
        n_val = max(1, int(self.holdout * len(idx)))
        va, tr = idx[:n_val], idx[n_val:]
        Xt, Xv = torch.from_numpy(X[tr]), torch.from_numpy(X[va])
        self.model = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        best, best_state, bad, hist = float("inf"), None, 0, []
        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch_size):
                xb = Xt[perm[i: i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model["dec"](self.model["enc"](xb)), xb)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = float(loss_fn(self.model["dec"](self.model["enc"](Xv)), Xv))
            hist.append(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.info = {"architecture": [int(X.shape[1]), self.hidden, self.code, self.hidden, int(X.shape[1])],
                     "activation": "ReLU (linear output)", "loss": "MSE", "optimizer": f"Adam(lr={self.lr})",
                     "batch_size": self.batch_size, "epochs_run": len(hist), "max_epochs": self.max_epochs,
                     "patience": self.patience, "holdout": self.holdout, "best_val_mse": best,
                     "train_rows": int(len(tr)), "val_rows": int(len(va)), "seed": self.seed}
        return self

    def transform(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        import torch

        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -20.0, 20.0)
        out = np.empty((len(X), self.code), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out[i: i + batch] = self.model["enc"](torch.from_numpy(X[i: i + batch])).numpy()
        return out

    def reconstruction_error(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        import torch

        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -20.0, 20.0)
        err = np.empty(len(X), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i: i + batch])
                err[i: i + batch] = ((self.model["dec"](self.model["enc"](xb)) - xb) ** 2).mean(dim=1).numpy()
        return err

    @property
    def names(self) -> list[str]:
        return [f"z_{i:02d}" for i in range(self.code)]

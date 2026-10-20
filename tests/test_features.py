"""Host-level temporal features: causality and agreement with a brute-force reference."""
import math

import numpy as np

from twws import features as F


def test_statistical_features_are_finite_where_defined(synth_flows):
    S = F.statistical_features(synth_flows)
    assert list(S.columns) == F.STAT_FEATURES
    A = S.to_numpy()
    assert not np.isinf(A).any()
    # undefined values are NaN and occur only where a quantity has no definition:
    # rates of zero-duration flows, the byte ratio of empty connections, entropy without payload
    may_be_nan = {"s_pkts_per_s", "s_bytes_per_s", "s_byte_ratio", "s_entropy"}
    has_nan = {c for i, c in enumerate(S.columns) if np.isnan(A[:, i]).any()}
    assert has_nan <= may_be_nan
    zero_dur = synth_flows["duration"].to_numpy() == 0
    assert np.isnan(S["s_pkts_per_s"].to_numpy()[zero_dur]).all()
    assert np.isfinite(S["s_pkts_per_s"].to_numpy()[~zero_dur]).all()


def test_temporal_features_match_brute_force(synth_flows):
    fl = synth_flows
    X, aux = F.build_base_features(fl)
    t = F.availability_time(fl)
    src = fl.src_ip.astype(str).to_numpy()
    dst = fl.dst_ip.astype(str).to_numpy()
    by, pk, du = fl.bytes.to_numpy(float), fl.pkts.to_numpy(float), fl.duration.to_numpy(float)
    iso = (fl.dir.to_numpy() == 0).astype(float)
    nr = (fl.rev_pkts.to_numpy() == 0).astype(float)
    dp = fl.dst_port.to_numpy()
    rng = np.random.default_rng(0)
    by_host = {h: np.flatnonzero(src == h) for h in np.unique(src)}
    worst = 0.0
    for i in rng.choice(len(fl), 150, replace=False):
        cand = by_host[src[i]]
        for wn, W in F.WINDOWS.items():
            m = cand[(t[cand] < t[i]) & (t[cand] >= t[i] - W)]      # strictly earlier records only
            n = len(m)
            ref = np.full(len(F.TEMPORAL_AGGS), np.nan)
            ref[0] = n
            if n:
                lb, lp, ld = np.log1p(by[m]), np.log1p(pk[m]), np.log1p(du[m])
                ref[1], ref[2] = math.log1p(by[m].sum()), math.log1p(pk[m].sum())
                ref[3], ref[4] = lb.mean(), (lb.std() if n > 1 else 0.0)
                ref[5], ref[6] = lp.mean(), (lp.std() if n > 1 else 0.0)
                ref[7], ref[8] = ld.mean(), (ld.std() if n > 1 else 0.0)
                ref[9], ref[10] = len(set(dst[m])), len(set(dp[m]))
                _, c = np.unique(dp[m], return_counts=True)
                p = c / c.sum()
                ref[11] = max(0.0, -(p * np.log(p)).sum())
                ref[12], ref[13] = iso[m].mean(), nr[m].mean()
            else:
                ref[[1, 2, 9, 10]] = 0.0
            got = X.iloc[i][[f"t_{wn}_{a}" for a in F.TEMPORAL_AGGS]].to_numpy(float)
            both_nan = np.isnan(ref) & np.isnan(got)
            assert not (np.isnan(ref) ^ np.isnan(got)).any()
            worst = max(worst, float(np.where(both_nan, 0.0, np.abs(ref - got)).max()))
    assert worst < 1e-5


def test_features_of_a_flow_do_not_depend_on_later_flows(synth_flows):
    """Dropping every record that becomes available later must not change a row's features."""
    fl = synth_flows.copy()
    t = F.availability_time(fl)
    cut = np.quantile(t, 0.5)
    early = fl[t <= cut].reset_index(drop=True)
    X_all, _ = F.build_base_features(fl)
    X_early, _ = F.build_base_features(early)
    A = X_all[(t <= cut)].to_numpy(float)
    B = X_early.to_numpy(float)
    same = np.isclose(A, B, atol=1e-6) | (np.isnan(A) & np.isnan(B))
    assert same.all()

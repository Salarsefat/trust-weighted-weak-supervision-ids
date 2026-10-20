"""The streaming feature engine reproduces the offline features record by record."""
import numpy as np

from twws import features as F
from twws.online import OnlineFeatureEngine


def test_engine_equals_offline_features(synth_flows):
    fl = synth_flows.copy()
    fl["t_avail"] = F.availability_time(fl)
    X, _ = F.build_base_features(fl)
    order = np.lexsort((np.arange(len(fl)), fl["t_avail"].to_numpy()))
    st = fl.iloc[order].reset_index(drop=True)
    for c in ("src_ip", "dst_ip"):
        st[c] = st[c].astype(str)
    eng = OnlineFeatureEngine(prune_every=64)
    H = eng.batch(st)
    ref = X[eng.names].to_numpy(dtype=np.float64)[order]
    assert not (np.isnan(H) ^ np.isnan(ref)).any()
    assert np.nanmax(np.where(np.isnan(H), 0.0, np.abs(H - ref))) < 1e-5

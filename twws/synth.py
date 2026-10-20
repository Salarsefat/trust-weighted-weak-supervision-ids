"""Synthetic flow generator -- FOR CODE VALIDATION ONLY.

The generator produces flow tables with the same schema as the Zeek extraction so that the
whole pipeline can be exercised end-to-end without the dataset (unit tests, smoke run, CI).
Nothing produced from synthetic flows is reported in the manuscript.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .extract import FLOW_COLUMNS
from .labels import load_schedule

BENIGN_SERVERS_EXT = [f"93.184.{i}.{j}" for i in range(210, 214) for j in range(10, 40)]
DNS_SERVER = "172.31.0.2"


def _conn_frame(n):
    return {k: np.zeros(n) for k in ["ts", "dur", "o_pk", "o_by", "r_pk", "r_by", "o_syn", "r_syn", "o_fin", "r_fin",
                                     "o_rst", "r_rst", "o_psh", "r_psh", "o_ack", "r_ack", "o_pay", "r_pay", "o_ent",
                                     "r_ent", "o_dur_frac", "r_dur_frac", "tls", "selfs", "alert"]}


def _tcp_conn(rng, n, o_pk, r_pk, o_bpp, r_bpp, dur, ent_o=4.5, ent_r=5.0, complete=1.0):
    c = _conn_frame(n)
    c["o_pk"] = np.maximum(1, rng.poisson(o_pk, n)).astype(float)
    done = rng.random(n) < complete
    c["r_pk"] = np.where(done, np.maximum(1, rng.poisson(r_pk, n)), 0).astype(float)
    c["o_pay"] = np.maximum(0, (c["o_pk"] - 3)) * np.maximum(rng.normal(o_bpp, o_bpp * 0.25, n), 0)
    c["r_pay"] = np.maximum(0, (c["r_pk"] - 2)) * np.maximum(rng.normal(r_bpp, r_bpp * 0.25, n), 0)
    c["o_by"] = c["o_pk"] * 40 + c["o_pay"]
    c["r_by"] = c["r_pk"] * 40 + c["r_pay"]
    c["dur"] = np.maximum(rng.gamma(2.0, dur / 2.0, n), 1e-4)
    c["o_syn"] = 1.0
    c["r_syn"] = np.where(done, 1.0, 0.0)
    c["o_fin"] = np.where(done, 1.0, 0.0)
    c["r_fin"] = np.where(done, 1.0, 0.0)
    c["o_ack"] = np.maximum(c["o_pk"] - 1, 0)
    c["r_ack"] = c["r_pk"]
    c["o_psh"] = np.floor(np.maximum(c["o_pk"] - 3, 0) * 0.6)
    c["r_psh"] = np.floor(np.maximum(c["r_pk"] - 2, 0) * 0.6)
    c["o_ent"] = np.where(c["o_pay"] > 0, np.clip(rng.normal(ent_o, 0.4, n), 0, 8), np.nan)
    c["r_ent"] = np.where(c["r_pay"] > 0, np.clip(rng.normal(ent_r, 0.4, n), 0, 8), np.nan)
    c["o_dur_frac"] = rng.uniform(0.85, 1.0, n)
    c["r_dur_frac"] = rng.uniform(0.80, 1.0, n)
    return c


def _udp_conn(rng, n, o_pk, r_pk, o_bpp, r_bpp, dur, ent=4.0):
    c = _conn_frame(n)
    c["o_pk"] = np.maximum(1, rng.poisson(o_pk, n)).astype(float)
    c["r_pk"] = rng.poisson(r_pk, n).astype(float)
    c["o_pay"] = c["o_pk"] * np.maximum(rng.normal(o_bpp, o_bpp * 0.2, n), 1)
    c["r_pay"] = c["r_pk"] * np.maximum(rng.normal(r_bpp, r_bpp * 0.2, n), 1)
    c["o_by"] = c["o_pk"] * 28 + c["o_pay"]
    c["r_by"] = c["r_pk"] * 28 + c["r_pay"]
    c["dur"] = np.maximum(rng.gamma(1.5, dur / 1.5, n), 1e-5)
    c["o_ent"] = np.clip(rng.normal(ent, 0.5, n), 0, 8)
    c["r_ent"] = np.where(c["r_pk"] > 0, np.clip(rng.normal(ent, 0.5, n), 0, 8), np.nan)
    c["o_dur_frac"] = 1.0
    c["r_dur_frac"] = 0.9
    return c


def _expand(c, orig_ip, resp_ip, orig_port, resp_port, proto, day, uid_prefix, start_uid):
    """Connection-level arrays -> two unidirectional flow rows per connection."""
    n = len(c["ts"])
    c = {k: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy() for k, v in c.items()}
    uid = np.array([f"{uid_prefix}{start_uid + i:09d}" for i in range(n)], dtype=object)
    state = np.where(c["r_pk"] > 0, "SF", "S0")
    rows = []
    for d, (a, b) in enumerate((("o", "r"), ("r", "o"))):
        A, B = (orig_ip, resp_ip) if d == 0 else (resp_ip, orig_ip)
        Ap, Bp = (orig_port, resp_port) if d == 0 else (resp_port, orig_port)
        rows.append(pd.DataFrame({
            "uid": uid, "dir": np.int8(d), "ts": c["ts"], "t_first": c["ts"] + (0.0 if d == 0 else 0.002),
            "t_last": c["ts"] + c["dur"] * c[f"{a}_dur_frac"],
            "duration": c["dur"] * c[f"{a}_dur_frac"] - (0.0 if d == 0 else 0.002), "conn_duration": c["dur"],
            "src_ip": A, "dst_ip": B, "src_port": np.asarray(Ap, dtype=np.int32), "dst_port": np.asarray(Bp, dtype=np.int32),
            "orig_ip": orig_ip, "resp_ip": resp_ip, "proto": proto,
            "pkts": c[f"{a}_pk"].astype(np.int64), "bytes": c[f"{a}_by"].astype(np.int64),
            "rev_pkts": c[f"{b}_pk"].astype(np.int64), "rev_bytes": c[f"{b}_by"].astype(np.int64),
            "syn": c[f"{a}_syn"].astype(np.int32), "fin": c[f"{a}_fin"].astype(np.int32),
            "rst": c[f"{a}_rst"].astype(np.int32), "psh": c[f"{a}_psh"].astype(np.int32),
            "ack": c[f"{a}_ack"].astype(np.int32), "urg": np.zeros(n, dtype=np.int32),
            "pay": c[f"{a}_pay"].astype(np.int64), "cap": c[f"{a}_pay"].astype(np.int64),
            "entropy": c[f"{a}_ent"].astype(np.float32), "ent_n": np.minimum(c[f"{a}_pay"], 4096).astype(np.int32),
            "conn_state": state, "tls": c["tls"].astype(np.int8),
            "tls_self_signed": np.where(c["tls"] > 0, c["selfs"], np.nan).astype(np.float32),
            "day": day, "snort_alert": c["alert"].astype(bool),
        }))
    out = pd.concat(rows, ignore_index=True)
    out["duration"] = out["duration"].clip(lower=0.0)
    return out[out["pkts"] > 0]


def make_synthetic_flows(schedule_path: str, n_benign_per_day: int = 20000, attack_scale: float = 1.0,
                         n_hosts: int = 60, seed: int = 0, days: list[str] | None = None,
                         utc_offset_hours: int = -4) -> pd.DataFrame:
    """Flows for the scheduled days: benign background plus one attack family per day."""
    rng = np.random.default_rng(seed)
    sched = load_schedule(schedule_path)
    hosts = [f"172.31.{64 + i // 200}.{10 + i % 200}" for i in range(n_hosts)]
    parts, uid0 = [], 0
    for day in (days or list(sched["days"])):
        d = sched["days"][day]
        t0 = d["midnight_utc"] - utc_offset_hours * 3600 + 8 * 3600      # 08:00 local
        span = 9 * 3600.0
        victims = d.get("victims_internal", [])
        day_hosts = sorted(set(hosts) | set(victims))
        # ---- benign: web (TLS), plain HTTP, DNS, internal chatter
        n = n_benign_per_day
        kinds = rng.choice(4, size=n, p=[0.45, 0.15, 0.30, 0.10])
        for kind in range(4):
            m = int((kinds == kind).sum())
            if m == 0:
                continue
            ts = t0 + np.sort(rng.uniform(0, span, m))
            src = rng.choice(day_hosts, m)
            if kind == 0:
                c = _tcp_conn(rng, m, 14, 18, 300, 900, 4.0, ent_o=7.3, ent_r=7.8)
                c["tls"] = 1.0
                dst, dport = rng.choice(BENIGN_SERVERS_EXT, m), np.full(m, 443)
                proto = "tcp"
            elif kind == 1:
                c = _tcp_conn(rng, m, 8, 10, 350, 1100, 2.0, ent_o=4.6, ent_r=5.6)
                dst, dport = rng.choice(BENIGN_SERVERS_EXT, m), np.full(m, 80)
                proto = "tcp"
            elif kind == 2:
                c = _udp_conn(rng, m, 1, 1, 40, 110, 0.03)
                dst, dport = np.full(m, DNS_SERVER, dtype=object), np.full(m, 53)
                proto = "udp"
            else:
                c = _tcp_conn(rng, m, 10, 10, 180, 240, 1.5, complete=0.97)
                dst, dport = rng.choice(day_hosts, m), rng.choice([445, 389, 135, 3389], m)
                proto = "tcp"
            c["ts"] = ts
            c["alert"] = (rng.random(m) < 0.004).astype(float)
            parts.append(_expand(c, src, dst, rng.integers(49152, 65535, m), dport, proto, day, "S", uid0))
            uid0 += m
        # ---- attacks of the day
        fam = d.get("family")
        for a in d.get("attacks", []):
            w0 = d["midnight_utc"] - utc_offset_hours * 3600 + a["start_s"]
            w1 = d["midnight_utc"] - utc_offset_hours * 3600 + a["end_s"]
            attackers = [ip for ip in a["attackers"] if not ip.startswith("172.31.")] or a["attackers"]
            base = {"BruteForce": 2500, "DoS": 4000, "DDoS": 6000, "Web": 250, "Infiltration": 120, "Bot": 1500}[fam]
            m = max(10, int(base * attack_scale))
            ts = np.sort(rng.uniform(w0, w1, m))
            if fam == "BruteForce":
                c = _tcp_conn(rng, m, 18, 20, 45, 60, 1.6, ent_o=5.5, ent_r=5.8)
                src, dst = rng.choice(attackers, m), rng.choice(victims, m)
                dport, proto, p_alert = np.full(m, 21 if "FTP" in a["name"] else 22), "tcp", 0.55
            elif fam == "DoS":
                slow = "Slow" in a["name"]
                c = _tcp_conn(rng, m, 5 if slow else 7, 3 if slow else 6, 120, 200, 40.0 if slow else 0.4,
                              complete=0.7)
                src, dst = rng.choice(attackers, m), rng.choice(victims, m)
                dport, proto, p_alert = np.full(m, 80), "tcp", 0.65
            elif fam == "DDoS":
                if "UDP" in a["name"]:
                    c = _udp_conn(rng, m, 400, 0, 32, 1, 20.0, ent=1.5)
                    proto = "udp"
                else:
                    c = _tcp_conn(rng, m, 6, 4, 90, 150, 0.3, complete=0.75)
                    proto = "tcp"
                src, dst = rng.choice(attackers, m), rng.choice(victims, m)
                dport, p_alert = np.full(m, 80), 0.45
            elif fam == "Web":
                c = _tcp_conn(rng, m, 9, 10, 420, 1000, 1.5, ent_o=4.9, ent_r=5.5)
                src, dst = rng.choice(attackers, m), rng.choice(victims, m)
                dport, proto, p_alert = np.full(m, 80), "tcp", 0.5
            elif fam == "Infiltration":
                c = _tcp_conn(rng, m, 30, 28, 200, 260, 60.0, ent_o=7.0, ent_r=7.0)
                src, dst = rng.choice(attackers, m), rng.choice(victims, m)
                dport, proto, p_alert = rng.choice([4444, 8443, 443], m), "tcp", 0.08
            else:  # Bot: bots call the command-and-control server
                c = _tcp_conn(rng, m, 6, 5, 90, 70, 0.8, ent_o=6.8, ent_r=6.6)
                c["tls"] = (rng.random(m) < 0.5).astype(float)
                c["selfs"] = 1.0
                src, dst = rng.choice(victims, m), rng.choice(attackers, m)
                dport, proto, p_alert = np.full(m, 8080), "tcp", 0.25
            c["ts"] = ts
            c["alert"] = (rng.random(m) < p_alert).astype(float)
            parts.append(_expand(c, src, dst, rng.integers(32768, 60999, m), dport, proto, day, "S", uid0))
            uid0 += m
            if d.get("ambiguous_rule") and victims:   # second-stage scan by the compromised host
                k = max(20, m // 2)
                c = _tcp_conn(rng, k, 2, 1, 1, 1, 0.01, complete=0.4)
                c["ts"] = np.sort(rng.uniform(w0, w1, k))
                parts.append(_expand(c, rng.choice(victims, k), rng.choice(day_hosts, k), rng.integers(32768, 60999, k),
                                     rng.integers(1, 1024, k), "tcp", day, "S", uid0))
                uid0 += k
    flows = pd.concat(parts, ignore_index=True)
    flows["capture"] = "synthetic"
    flows["cap_id"] = np.int32(0)
    for col in ["src_ip", "dst_ip", "orig_ip", "resp_ip", "proto", "conn_state", "day", "capture"]:
        flows[col] = flows[col].astype(str).astype("category")
    return flows[FLOW_COLUMNS + ["day", "capture", "cap_id", "snort_alert"]].sort_values("ts", kind="stable").reset_index(drop=True)

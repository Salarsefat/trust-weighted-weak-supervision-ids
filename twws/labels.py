"""Schedule-based ground truth.

Ground-truth labels are used for *evaluation only*.  They are derived from the published
attack schedules (attacker addresses, victim addresses, attack windows), which is also how
the dataset providers label their own flow files.  The schedules are transcribed in
``configs/attack_schedule_ids2018.yaml`` and ``configs/attack_schedule_ids2017.yaml``.

CSE-CIC-IDS2018 (one attack family per capture day)
    A flow is malicious, with the family of its capture day, when one of its endpoints is an
    attacker address listed for that day.  Time windows only name the attack within the day.
    On the two infiltration days, connections *originated by the compromised victim* towards
    internal addresses during an attack window are marked ``Ambiguous`` (the schedule cannot
    tell second-stage scanning from the victim's ordinary traffic) and are excluded from
    training and evaluation rows, although they stay in the host streams used for the
    temporal features.

CICIDS2017 (single capture point in front of and behind the firewall NAT)
    A flow is malicious when (a) an endpoint is an attacker address, or (b) it runs between
    the NAT address and a victim's local address inside an attack window.
"""
from __future__ import annotations

import calendar
import ipaddress
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import REPO_ROOT
from .utils import get_logger

log = get_logger("twws.labels")

BENIGN, AMBIGUOUS = "Benign", "Ambiguous"


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 3600 + int(m) * 60


def load_schedule(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as fh:
        sched = yaml.safe_load(fh)
    for day, d in sched["days"].items():
        d["midnight_utc"] = calendar.timegm(datetime.strptime(d["date"], "%Y-%m-%d").timetuple())
        for a in d.get("attacks", []):
            a["start_s"], a["end_s"] = _hhmm(a["start"]), _hhmm(a["end"])
            a.setdefault("family", d.get("family"))
        d["attackers"] = sorted({ip for a in d.get("attacks", []) for ip in a.get("attackers", [])})
    return sched


def _local_tod(ts: np.ndarray, midnight_utc: float, offset_hours: float) -> np.ndarray:
    """Seconds since local midnight of the capture day."""
    return ts + offset_hours * 3600.0 - midnight_utc


def _in_net(ips: pd.Series, net: str) -> np.ndarray:
    """Vectorised membership of dotted-quad strings in an IPv4 network (prefix match on ints)."""
    network = ipaddress.ip_network(net)
    uniq = pd.unique(ips)
    table = {}
    for u in uniq:
        try:
            table[u] = ipaddress.ip_address(u) in network
        except ValueError:
            table[u] = False
    return ips.map(table).to_numpy(dtype=bool)


def _attacker_mask(df: pd.DataFrame, ips: list[str]) -> np.ndarray:
    s = set(ips)
    return (df["src_ip"].isin(s) | df["dst_ip"].isin(s)).to_numpy()


def _day_rows(flows: pd.DataFrame, day: str) -> np.ndarray:
    """Positions of the rows of one capture day (on the category codes when the column is categorical,
    so that no per-row string object is created for a stream of tens of millions of flows)."""
    col = flows["day"]
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = col.cat.categories
        if day not in cats:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(col.cat.codes.to_numpy() == cats.get_loc(day))
    return np.flatnonzero(col.to_numpy() == day)


def estimate_utc_offset(flows: pd.DataFrame, sched: dict, candidates=range(-12, 13)) -> tuple[int, pd.DataFrame]:
    """Choose the UTC offset under which attacker traffic best falls inside the scheduled windows.

    Returns the best integer-hour offset and, per candidate, the share of attacker-involved
    flows inside a window (with the schedule's slack) - a diagnostic that is written to disk.
    """
    slack = float(sched.get("window_slack_seconds", 300))
    rows = []
    global_att = sched.get("attacker_addresses", [])
    parts = []
    for day, d in sched["days"].items():
        if not d.get("attacks"):
            continue
        sub = flows[["src_ip", "dst_ip", "ts"]].iloc[_day_rows(flows, day)]
        if sub.empty:
            continue
        att = sorted(set(d["attackers"]) | set(global_att))
        m = _attacker_mask(sub, att)
        if m.any():
            parts.append((d, sub.loc[m, "ts"].to_numpy(dtype=float)))
    for off in candidates:
        inside = total = 0
        for d, ts in parts:
            tod = _local_tod(ts, d["midnight_utc"], off)
            hit = np.zeros(len(ts), dtype=bool)
            for a in d["attacks"]:
                hit |= (tod >= a["start_s"] - slack) & (tod <= a["end_s"] + slack)
            inside += int(hit.sum())
            total += len(ts)
        rows.append({"utc_offset_hours": off, "attacker_flows": total, "inside_window": inside,
                     "share_inside": inside / total if total else np.nan})
    diag = pd.DataFrame(rows)
    if diag["attacker_flows"].max() == 0 or diag["share_inside"].isna().all():
        best = int(sched.get("default_utc_offset_hours", 0))
        log.warning("no attacker traffic found; keeping default UTC offset %+d h", best)
    else:
        top = diag["share_inside"].max()
        cand = diag[np.isclose(diag["share_inside"], top)]["utc_offset_hours"].tolist()
        default = int(sched.get("default_utc_offset_hours", 0))
        best = default if default in cand else int(cand[0])
        log.info("estimated UTC offset %+d h (%.1f%% of attacker flows inside scheduled windows)", best, 100 * top)
        if top < 0.5:
            log.warning("at most %.1f%% of the attacker flows fall inside the scheduled attack windows for any UTC "
                        "offset: check the schedule, the capture clock and the day assignment before using the labels",
                        100 * top)
    return best, diag


# how a flow involved in several attacks of one day is named (recorded in stream_info.json, so that a corpus built with
# an earlier rule can be recognised): "victim-first" = the attack whose victim it reaches, then the nearest window
FAMILY_RULE = "victim-first"


def label_flows(flows: pd.DataFrame, sched: dict, offset_hours: float | None = None) -> pd.DataFrame:
    """Return a frame aligned with ``flows`` holding ``family``, ``attack``, ``label``, ``in_window``.

    ``flows`` needs the columns day, ts, src_ip, dst_ip, orig_ip, resp_ip.
    """
    if offset_hours is None:
        offset_hours = sched.get("default_utc_offset_hours", 0)
    slack = float(sched.get("window_slack_seconds", 300))
    n = len(flows)
    family = np.full(n, BENIGN, dtype=object)
    attack = np.full(n, "", dtype=object)
    in_window = np.zeros(n, dtype=bool)
    ts_all = flows["ts"].to_numpy(dtype=float)
    global_att = sched.get("attacker_addresses", [])
    nat = sched.get("nat_address")
    internal = sched.get("internal_net")

    addr_cols = [c for c in ("src_ip", "dst_ip", "orig_ip", "resp_ip") if c in flows.columns]
    for day, d in sched["days"].items():
        idx = _day_rows(flows, day)
        if idx.size == 0 or not d.get("attacks"):
            continue
        sub = flows[addr_cols].iloc[idx]          # only the address columns are read below
        tod = _local_tod(ts_all[idx], d["midnight_utc"], offset_hours)
        fam = np.full(idx.size, BENIGN, dtype=object)
        att_name = np.full(idx.size, "", dtype=object)
        inwin = np.zeros(idx.size, dtype=bool)
        best_dist = np.full(idx.size, np.inf)
        best_victim = np.zeros(idx.size, dtype=bool)

        for a in d["attacks"]:
            win = (tod >= a["start_s"] - slack) & (tod <= a["end_s"] + slack)
            # distance to the window, used to name attacker flows that fall outside every window
            dist = np.where(win, 0.0, np.minimum(np.abs(tod - a["start_s"]), np.abs(tod - a["end_s"])))
            involved = _attacker_mask(sub, a.get("attackers", []))
            victim = np.zeros(idx.size, dtype=bool)
            if a.get("victims"):
                v = set(a["victims"])
                victim = (sub["src_ip"].isin(v) | sub["dst_ip"].isin(v)).to_numpy()
                if nat:
                    pair = ((sub["src_ip"] == nat) & sub["dst_ip"].isin(v)) | ((sub["dst_ip"] == nat) & sub["src_ip"].isin(v))
                    involved = involved | (pair.to_numpy() & win)
            # A flow involved in several attacks of the day is named after the attack whose victim it reaches, and
            # among those (or if none lists it as a victim) after the attack with the nearest window.  Without the
            # victim rule, the bots of CICIDS2017, which contact the attacker all day, would be named after
            # whichever other attack of the same attacker lies nearer in time.
            take = involved & ((victim & ~best_victim) | ((victim == best_victim) & (dist < best_dist)))
            fam[take] = a.get("family") or d.get("family")
            att_name[take] = a["name"]
            inwin[take] = win[take]
            best_dist[take] = dist[take]
            best_victim[take] = victim[take]

            # ambiguous second-stage traffic (connection originated by a compromised host)
            amb_orig = a.get("ambiguous_originators")
            if amb_orig is None and d.get("ambiguous_rule") == "victim_originated_internal_in_window":
                amb_orig = d.get("victims_internal", [])
            if amb_orig and internal:
                amb = sub["orig_ip"].isin(set(amb_orig)).to_numpy() & _in_net(sub["resp_ip"], internal) & win
                amb &= fam == BENIGN
                fam[amb] = AMBIGUOUS
                att_name[amb] = a["name"] + ":second-stage"

        if global_att:  # attacker addresses outside every listed attack of the day
            extra = _attacker_mask(sub, global_att) & (fam == BENIGN)
            fam[extra] = d.get("family") or "Attack"
            att_name[extra] = "attacker-address"
        family[idx], attack[idx], in_window[idx] = fam, att_name, inwin

    out = pd.DataFrame({"family": family, "attack": attack, "in_window": in_window}, index=flows.index)
    out["label"] = np.where(out["family"] == BENIGN, 0, np.where(out["family"] == AMBIGUOUS, -1, 1)).astype(np.int8)
    return out


def label_summary(flows: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Per day and family: number of flows and share of malicious flows inside a scheduled window."""
    # string arrays as they are (no per-row Python objects); the day as strings, as before
    t = pd.DataFrame({"day": flows["day"].astype(str).array, "family": lab["family"].astype(str).array,
                      "attack": lab["attack"].astype(str).array, "in_window": lab["in_window"].to_numpy()})
    g = t.groupby(["day", "family", "attack"], sort=True)
    return g.agg(flows=("in_window", "size"), inside_window=("in_window", "sum")).reset_index()

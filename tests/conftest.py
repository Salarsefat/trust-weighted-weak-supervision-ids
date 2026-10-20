"""Shared fixtures.  All tests run on synthetic flows or on tiny captures generated on the fly."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEDULE = str(REPO / "configs" / "attack_schedule_ids2018.yaml")


def have(tool: str) -> bool:
    from twws.utils import _find_tool

    exe = _find_tool(tool, ("/opt/zeek/bin", "/usr/local/zeek/bin", "/usr/sbin", "/usr/local/bin"))
    return shutil.which(exe) is not None or Path(exe).exists()


@pytest.fixture(scope="session")
def synth_flows():
    from twws.synth import make_synthetic_flows

    return make_synthetic_flows(SCHEDULE, n_benign_per_day=2500, attack_scale=0.2, n_hosts=20, seed=1)


@pytest.fixture(scope="session")
def schedule():
    from twws.labels import load_schedule

    return load_schedule(SCHEDULE)

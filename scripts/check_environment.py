#!/usr/bin/env python3
"""Report what the pipeline needs and what this machine provides."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from twws.utils import environment_record

    env = environment_record()
    disk = shutil.disk_usage(os.getcwd())
    env["free_disk_gb"] = round(disk.free / 1e9, 1)
    print(json.dumps(env, indent=2))
    problems = []
    if sys.version_info < (3, 10):
        problems.append("Python >= 3.10 is required")
    missing = [k for k, v in env["packages"].items() if v is None]
    if missing:
        problems.append("missing Python packages: " + ", ".join(missing) + "  (pip install -r requirements.txt)")
    if not env["tools"].get("zeek"):
        problems.append("zeek not found (needed for extraction; set tools.zeek in the configuration if it is not on PATH)")
    if not env["tools"].get("snort"):
        problems.append("snort not found (needed for the signature oracle only)")
    if (env.get("memory_gb") or 0) < 30:
        problems.append("less than 32 GB of memory: use configs/paper_reduced.yaml")
    print("\n".join("WARNING: " + p for p in problems) if problems else "\nEnvironment looks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

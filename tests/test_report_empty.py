"""The report should identify missing results without external datasets."""
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _report_without_data(tmp_path, out):
    """``twws.cli report`` on empty workspaces only: every directory from which the report reads data is redirected
    into ``tmp_path``, so that the test does not depend on what the machine running it holds (on a machine with the
    captures, ``work/ids2018`` and ``work/ids2017`` exist and the report would find their files)."""
    return subprocess.run([sys.executable, "-m", "twws.cli", "report", "--config", str(REPO / "configs" / "paper.yaml"),
                           "--set", f"work_dir={tmp_path / 'work'}", "--set", f"capture_dir={tmp_path / 'captures'}",
                           "--set", f"transfer.work_dir={tmp_path / 'second'}",
                           "--set", f"transfer.model_path={tmp_path / 'transfer_rf.joblib'}", "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True)


def test_report_without_results_names_every_table(tmp_path):
    """Before any experiment has run, the report holds one named placeholder per table and no result."""
    from twws.report.build import TABLE_PURPOSE

    out = tmp_path / "generated"
    r = _report_without_data(tmp_path, out)
    assert r.returncode == 0, r.stderr[-3000:]
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))["text"]
    assert numbers["results.available"] == ""
    assert not any(k.startswith(("main.", "agg.", "base.", "online.", "cross.f")) for k in numbers)   # no result key
    for name, purpose in TABLE_PURPOSE.items():
        tex = (out / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        assert "RESULT NOT AVAILABLE" in tex, name
        assert f"\\label{{tab:{name}}}" in tex, name
        caption = re.search(r"\\caption\{\\tabtag\{[^}]*\}(.*)\}\\label", tex).group(1)
        # the caption of a placeholder is the purpose of that table, never the caption of another one
        assert caption.replace("\\", "") == purpose.replace("\\", "") or purpose[:25] in caption, name


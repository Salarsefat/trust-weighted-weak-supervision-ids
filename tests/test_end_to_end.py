"""Whole pipeline on synthetic flows: stages, two experiments and the report (about a minute)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_cli_pipeline_and_report(tmp_path):
    work = tmp_path / "work"
    common = ["--config", str(REPO / "configs" / "smoke.yaml"), "--set", f"work_dir={work}",
              "--set", f"transfer.work_dir={tmp_path / 'src'}", "--set", "synthetic.n_benign_per_day=3000",
              "--set", "corpus.family_cap=1200", "--set", "experiments.seeds=[7,13]",
              "--set", "experiments.pipeline_seeds=[]", "--set", "search.n_trials=3", "--set", "search.n_random=2"]

    def cli(*args):
        r = subprocess.run([sys.executable, "-m", "twws.cli", *args, *common], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        return r

    cli("synthetic")
    cli("label-check")                                   # weak-label diagnostic before the experiments
    chk = json.loads((work / "calibration" / "label_check.json").read_text())
    assert set(chk["aggregation_train"]) == {"majority", "ds"} and chk["train_rows"] > 0
    assert len(chk["per_oracle_train"]) == len(chk["oracles"]) >= 4
    cli("run", "search", "main", "--n-jobs", "2")
    main = json.loads((work / "results" / "main.json").read_text())
    assert main["synthetic"] is True
    assert main["summary"]["f1"]["mean"] > 0.6
    rep = main["representative"]
    assert sum(f["tp"] for f in rep["families"]) == rep["tp"]          # family table adds up to the confusion matrix
    assert sum(f["fn"] for f in rep["families"]) == rep["fn"]
    out = tmp_path / "generated"
    cli("report", "--out", str(out))
    numbers = (out / "numbers.tex").read_text()
    assert "twws@main.f1" in numbers and "twws@synthetic}{1}" in numbers
    assert "RESULT NOT AVAILABLE" in (out / "tables" / "baselines.tex").read_text()   # not run -> visible placeholder
    assert "RESULT NOT AVAILABLE" not in (out / "tables" / "main.tex").read_text()

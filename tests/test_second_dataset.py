"""The method applied to CICIDS2017 without its labels (configs/ids2017_ext.yaml; docs/ORACLES_EXTENDED.md, step 5)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, params_file):
    from twws.config import apply_dotted_overrides, load_config
    return apply_dotted_overrides(load_config(str(REPO / "configs" / "ids2017_ext.yaml")), [
        f"work_dir={tmp_path / 'w'}", f"corpus_work_dir={tmp_path / 'c'}", f"capture_dir={tmp_path / 'c'}",
        f"search.params_from={params_file}"])


def test_configuration_is_the_extended_method_without_the_transfer_oracle(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "search.json")
    assert list(cfg.oracles.enabled) == ["stat", "signature", "cluster", "domain", "rate", "beacon"]
    assert cfg.oracles.aggregation == "ds_grouped" and float(cfg.oracles.group_jaccard) == 0.5
    assert not bool(cfg.refinement.enabled)
    assert cfg.corpus.name == "CICIDS2017" and list(cfg.experiments.pipeline_seeds) == []
    # the oracle settings are those of the extended configuration on CSE-CIC-IDS2018
    from twws.config import load_config
    ext = load_config(str(REPO / "configs" / "paper_reduced_ext.yaml"))
    for k in ("rate", "beacon", "stat", "cluster", "domain"):
        assert dict(cfg.oracles[k]) == dict(ext.oracles[k]), k
    assert ext.report.second_work_dir == "work/ids2017_ext" and cfg.report.second_work_dir is None


def test_hyperparameters_come_from_the_search_of_another_workspace(tmp_path):
    from twws.experiments.common import Ctx
    p = tmp_path / "search.json"
    p.write_text(json.dumps({"best_params": {"eta": 0.123, "max_depth": 4, "class_weight": None}}), encoding="utf-8")
    ctx = Ctx(_cfg(tmp_path, p), n_jobs=1)
    bp = ctx.best_params()
    assert bp["eta"] == 0.123 and bp["max_depth"] == 4 and bp["class_weight"] == "none"
    assert ctx.params_source.startswith("search of another workspace")


def test_a_missing_parameter_file_is_an_error(tmp_path):
    from twws.experiments.common import Ctx
    with pytest.raises(RuntimeError, match="does not exist"):
        Ctx(_cfg(tmp_path, tmp_path / "missing.json"), n_jobs=1).best_params()


def test_the_configuration_fixed_before_testing_is_applied_as_specified():
    """configs/ids2017_ext_frozen.yaml: the same test for the configuration as it was specified (refinement on), with
    the hyperparameters selected for that configuration; everything else as in configs/ids2017_ext.yaml."""
    from twws.config import load_config
    fr = load_config(str(REPO / "configs" / "ids2017_ext_frozen.yaml"))
    sel = load_config(str(REPO / "configs" / "ids2017_ext.yaml"))
    assert bool(fr.refinement.enabled) and not bool(sel.refinement.enabled)
    assert fr.search.params_from == "work/ids2018_reduced_ext_frozen/results/search.json"
    assert fr.work_dir == "work/ids2017_ext_frozen" and fr.corpus_work_dir == sel.corpus_work_dir
    assert list(fr.oracles.enabled) == list(sel.oracles.enabled)
    for k in ("rate", "beacon", "stat", "cluster", "domain"):
        assert dict(fr.oracles[k]) == dict(sel.oracles[k]), k
    ext = load_config(str(REPO / "configs" / "paper_reduced_ext.yaml"))
    frozen = load_config(str(REPO / "configs" / "paper_reduced_ext_frozen.yaml"))
    assert dict(fr.refinement) == dict(frozen.refinement)
    assert ext.report.second_frozen_work_dir == "work/ids2017_ext_frozen" and fr.report.second_frozen_work_dir is None

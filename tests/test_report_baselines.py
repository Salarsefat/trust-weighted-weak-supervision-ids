"""Sentences on the end-to-end baselines that the report writes when one, several or no configured models are missing
(twws/report/build.py, b_baselines): the manuscript and the letter take them as they are."""
import statistics
from types import SimpleNamespace

import pytest

from twws.report.build import Numbers, b_baselines

SEEDS = [7, 13, 21]
ORDER = ["rf", "lstm", "cnn_lstm", "kitnet"]
NAMES = {"rf": "Random Forest", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "kitnet": "Kitsune (KitNET)"}
METRICS = ("precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc")


def _summary(xs):
    return {"mean": sum(xs) / len(xs), "std": statistics.stdev(xs), "ci": 0.001, "n": len(xs), "min": min(xs),
            "max": max(xs)}


def _result(f1s, paired=True):
    """Result of one model over the seeds, in the layout of the result files (runs and their summary)."""
    runs = [{"seed": s, "precision": 0.95, "recall": 0.87, "f1": f, "accuracy": 0.97, "fpr": 0.009, "auc_roc": 0.99,
             "pr_auc": 0.97 + i * 1e-3} for i, (s, f) in enumerate(zip(SEEDS, f1s))]
    if paired:
        runs[0]["paired_vs_proposed"] = {"diff_mean": 0.001, "lo": 0.0005, "hi": 0.0015, "p_two_sided": 0.0}
    summary = {k: _summary([r[k] for r in runs]) for k in METRICS}
    summary["families"] = {"Bot": {**_summary([0.51, 0.52, 0.5]), "support": 100}}
    return {"runs": runs, "summary": summary}


def _numbers(tmp_path, done):
    (tmp_path / "tables").mkdir()
    (tmp_path / "supplementary").mkdir()
    models = {k: {"name": NAMES[k], **_result([0.910, 0.911, 0.912])} for k in done}
    R = {"main": _result([0.912, 0.913, 0.911], paired=False),
         "baselines": {"models": models, "seeds": SEEDS, "protocol": {"trials": 30}}}
    cfg = SimpleNamespace(experiments=SimpleNamespace(baselines=ORDER))
    N = Numbers()
    b_baselines(cfg, None, R, N, tmp_path)
    return N.text


@pytest.mark.parametrize("done, missing", [(["rf", "cnn_lstm", "kitnet"], "LSTM"),
                                           (["rf", "kitnet"], "LSTM and CNN-LSTM")])
def test_sentences_name_the_missing_baselines(tmp_path, done, missing):
    T = _numbers(tmp_path, done)
    one = missing == "LSTM"
    verb = "was" if one else "were"
    assert T["base.compare_clause"] == f"{len(done)} of the 4 baselines"
    assert T["base.status_kinds_clause"].startswith(f"{missing} {verb} not completed in the time available; "
                                                    "the completed baselines comprise ")
    assert T["base.setup_sentence"].endswith(f"lists {missing} as not completed.")
    assert f"Decide whether to complete {'it' if one else 'them'} before submission" in T["base.open_item"]
    assert T["base.letter_missing_sentence"] == (f"Of the 4 baselines of the protocol, {missing} {verb} not completed "
                                                 "in the time available; the manuscript names any baseline that is "
                                                 "missing.")
    assert T["base.letter_constraint_clause"].startswith(f"of the 4 baselines of the comparison protocol, {missing} ")
    assert (tmp_path / "tables" / "baselines.tex").read_text().count("not completed in the time available") == len(
        missing.split(" and "))


def test_sentences_when_every_baseline_is_completed(tmp_path):
    T = _numbers(tmp_path, ORDER)
    assert T["base.compare_clause"] == "all 4 baselines"
    assert T["base.status_kinds_clause"] == ("the baselines comprise 1 tree ensemble, 2 neural networks trained with a "
                                             "supervised loss (LSTM and CNN-LSTM), and the autoencoder-based anomaly "
                                             "detector Kitsune (KitNET)")
    assert T["base.setup_sentence"] == "All 4 models were completed."
    assert T["base.completion_sentence"] == ""       # Section 7.7 then states no limitation on the completion
    assert T["base.open_item"] == ""          # the letter drops an empty open item
    assert T["base.letter_missing_sentence"] == "All 4 baselines of the protocol were completed."
    assert "not completed" not in (tmp_path / "tables" / "baselines.tex").read_text()


def test_ranking_clause_separates_intervals_that_exclude_zero():
    from twws.report.build import _ranking_seed_clause
    done = {k: {"name": NAMES[k]} for k in ("rf", "lstm", "cnn_lstm")}
    d = lambda mean, ci, hi, n=5: {"mean": mean, "ci": ci, "n_baseline_higher": hi, "n": n}
    diffs = {("lstm", "pr_auc"): d(-0.02, 0.01, 5), ("cnn_lstm", "pr_auc"): d(-0.01, 0.02, 4),
             ("lstm", "auc_roc"): d(-0.01, 0.02, 4), ("cnn_lstm", "auc_roc"): d(-0.01, 0.005, 5)}
    s, ex, zero, ex_auc = _ranking_seed_clause(done, ["lstm", "cnn_lstm"], ["lstm", "cnn_lstm"], diffs)
    assert (ex, zero, ex_auc) == (["lstm"], ["cnn_lstm"], ["cnn_lstm"])
    assert s == ("the 95\\% interval of the difference in PR-AUC excludes zero for LSTM, which ranks the test flows "
                 "better than the proposed detector on every seed, and it contains zero for CNN-LSTM; of these two "
                 "baselines, the interval of the difference in AUC-ROC excludes zero only for CNN-LSTM")


def test_seed_sentence_and_spread_clause_follow_the_outcome(tmp_path):
    """With every learner within the seed-to-seed range, the sentences say so; a learner that is worse on every seed
    is named with its interval, and a spread beyond the seed range is stated with its value."""
    T = _numbers(tmp_path, ["rf", "cnn_lstm"])
    assert T["base.sup.seed_sentence"].startswith("Matched by seed, the difference to each of these baselines has a "
                                                  "95\\% interval that contains zero")
    assert "changes the mean F1-score by less than the choice of the seed" in T["base.sup.spread_clause"]
    assert T["base.sup.seed_letter_clause"].startswith("matched by seed, the F1-score of none of them differs")
    worse = tmp_path / "worse"
    worse.mkdir()
    (worse / "tables").mkdir()
    (worse / "supplementary").mkdir()
    models = {"rf": {"name": NAMES["rf"], **_result([0.910, 0.911, 0.912])},
              "lstm": {"name": NAMES["lstm"], **_result([0.905, 0.906, 0.904])}}
    R = {"main": _result([0.912, 0.913, 0.911], paired=False),
         "baselines": {"models": models, "seeds": SEEDS, "protocol": {"trials": 30}}}
    N = Numbers()
    b_baselines(SimpleNamespace(experiments=SimpleNamespace(baselines=ORDER)), None, R, N, worse)
    s = N.text["base.sup.seed_sentence"]
    assert "except LSTM (+0.0070 [" in s and "better than this baseline on every seed" in s
    assert "advantage of the proposed detector over LSTM only" in s
    assert N.text["base.sup.seed_letter_clause"].startswith("matched by seed, the F1-score differs from that of the "
                                                            "proposed detector with a 95\\% interval that excludes "
                                                            "zero only for LSTM (+0.0070 [")
    assert N.text["base.sup.spread_clause"].startswith("the spread of their mean F1-scores (0.0060) is larger than")

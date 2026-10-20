"""Experiment registry: name -> callable(ctx) writing ``results/<name>.json``."""
from __future__ import annotations


def registry() -> dict:
    from . import core, diagnostics, extra

    return {
        "search": core.exp_search,
        "main": core.exp_main,
        "oracles": core.exp_oracles,
        "ablations": core.exp_ablations,
        "fs_compare": core.exp_fs_compare,
        "loao": core.exp_loao,
        "protocols": core.exp_protocols,
        "baselines": extra.exp_baselines,
        "adversarial": extra.exp_adversarial,
        "interpret": extra.exp_interpret,
        "online": extra.exp_online,
        "cross_dataset": extra.exp_cross_dataset,
        "qos": extra.exp_qos,
        # recomputes only the trust table of "oracles" (binning corrected in code version 25); not part of ORDER,
        # because "oracles" itself now writes the corrected table
        "trust": extra.exp_trust,
        # diagnostics of the labelling rules added after the results (oracle votes only; not part of ORDER)
        "diag_oracles": diagnostics.exp_diag_oracles,
        # sensitivity of the feature selection to d' and F (trains detectors; reference re-run in the same environment)
        "diag_selection": diagnostics.exp_diag_selection,
    }


# order used by ``twws run-all``; cheap and central experiments first
ORDER = ["search", "main", "oracles", "protocols", "loao", "ablations", "fs_compare", "interpret", "adversarial",
         "online", "qos", "baselines", "cross_dataset"]

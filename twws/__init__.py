"""Weakly supervised intrusion detection without labels from the monitored network (TWWS-IDS).

Reference implementation accompanying the manuscript
"Weakly Supervised Intrusion Detection Without Labels from the Monitored Network:
A Botnet-Focused Study in Cloud Environments" (the package name stems from an
earlier title, "Trust-Weighted Weak Supervision").

The package is organised by pipeline step:

    fetch        capture retrieval, file-level manifest and checksums
    extract      Zeek-based flow extraction (per-direction flow records)
    snort        signature oracle (Snort alerts mapped onto flows)
    labels       schedule-based ground truth (evaluation only)
    corpus       de-duplication, down-sampling and partition protocols
    features     statistical, causal temporal and autoencoder latent features
    preprocess   cluster-median imputation and robust scaling
    oracles      the five weak-supervision oracles
    weak         majority vote, trust scores and semi-supervised refinement
    fselect      stability-aware feature selection and selection baselines
    objective    trust-weighted focal objective for XGBoost
    model        training, early stopping and threshold calibration
    baselines    end-to-end baselines
    experiments  every experiment reported in the manuscript
    report       tables, figures and supplementary files generated from results
"""

__version__ = "1.0.0"

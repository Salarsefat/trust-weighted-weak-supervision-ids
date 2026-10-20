# Trust-Weighted Weak Supervision for Intrusion Detection

Reproduction source for *Trust-Weighted Weak Supervision for Label-Efficient Intrusion Detection in Cloud Environments: A Botnet-Focused Study* by S. S. Sefati et al. (Journal of Network and Computer Applications, manuscript under review).

The pipeline extracts network flows with Zeek, maps Snort alerts, combines seven weak labeling rules, and trains a trust-weighted XGBoost detector. It includes the experiment and report code used for the study. No dataset, trained model, result table, figure, or manuscript is distributed here.

## Requirements

- Linux or macOS, Python 3.10 or newer. The supplied dependency versions were validated with Python 3.13.
- Zeek 8.0.10 and Snort 2.9.20 for the capture-based experiments. A synthetic smoke run does not need either tool.
- Access to the CSE-CIC-IDS2018 and CICIDS2017 packet captures. These are large external datasets; see [data instructions](docs/DATA.md).
- Around 25 GB of usable memory for the paper's one-million-flow profile, plus substantial disk space for selected captures. Run `python scripts/check_environment.py` before downloading data.

## Install and check

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m twws.cli synthetic --config configs/smoke.yaml
python -m twws.cli run-all --config configs/smoke.yaml --n-jobs 2
```

The smoke configuration generates synthetic flows solely to check the pipeline. Its metrics are not study results. The test suite uses generated flows and a small capture fixture; tests requiring Zeek or Snort skip if those tools are unavailable.

## Reproduce the study

Start with [the complete command sequence](docs/REPRODUCE.md). The main profile is `configs/paper_reduced_ext.yaml`, built on the corpus configuration `configs/paper_reduced.yaml`. Source data, experiment output, generated reports, and local caches go into ignored `work/` directories.

```bash
python -m twws.cli run-all --config configs/paper_reduced_ext.yaml
python -m twws.cli report --config configs/paper_reduced_ext.yaml --out work/report
```

These two commands require the capture and transfer-model preparation described in the reproduction guide. The report generator may create figures and tables in `work/report`; none are tracked in this repository.

## Layout

| Path | Purpose |
| --- | --- |
| `twws/` | Retrieval, feature extraction, weak supervision, detector, experiments, and report source |
| `zeek/` | Flow extraction script |
| `configs/` | Study profiles, attack schedules, and the small pinned archive inventory needed for data selection |
| `scripts/` | Environment check and optional Zeek build instructions |
| `tests/` | Checks using generated data and one small capture fixture |
| `docs/` | Dataset access and reproduction commands |

The target dataset's ground-truth labels are used to evaluate results, not to fit the detector. The transfer rule is trained with labeled CICIDS2017 flows and excluded when evaluating on CICIDS2017.

## Use and citation

No license has been granted for reuse or redistribution. For access to the datasets, follow the dataset providers' terms. If you use the study, cite the article above; final bibliographic details can be added after publication.

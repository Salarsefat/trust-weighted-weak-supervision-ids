# Reproduce the experiments

Run commands from the repository root inside the environment described in [README](../README.md). Output is written below `work/`, which Git ignores. The real-data pipeline can take many hours and requires large external packet captures. The Python tests and synthetic run are checks, not substitutes for the datasets.

## 1. Prepare CICIDS2017

Download the five daily captures from the provider linked in [DATA.md](DATA.md), verify their provider MD5 files, and put them under one local directory. This corpus supplies the separately labeled transfer rule and the second-dataset evaluation.

```bash
python -m twws.cli fetch --config configs/ids2017.yaml --local-root /data/CICIDS2017
python -m twws.cli extract --config configs/ids2017.yaml
python -m twws.cli corpus --config configs/ids2017.yaml
```

## 2. Prepare CSE-CIC-IDS2018

The repository's pinned inventory and attack schedules select the captures; the raw data remain with the dataset provider. Inspect the selection before retrieval. `--delete-pcaps` removes local downloaded captures after extraction to limit disk use.

```bash
python -m twws.cli fetch --config configs/paper_reduced.yaml --list-only
python -m twws.cli ingest --config configs/paper_reduced.yaml --delete-pcaps
python -m twws.cli corpus --config configs/paper_reduced.yaml
python -m twws.cli transfer --config configs/paper_reduced.yaml
python -m twws.cli pairs --config configs/paper_reduced.yaml
```

If the captures are already stored locally, add `--local-root /path/to/ids2018` to `ingest`. See [DATA.md](DATA.md) for accepted layouts, capture verification, the labeling rule, and the sampling protocol.

## 3. Run the selected method

```bash
python -m twws.cli label-check --config configs/paper_reduced_ext.yaml
python -m twws.cli run-all --config configs/paper_reduced_ext.yaml
```

`run-all` starts with hyperparameter search, then runs the registered experiments. Completed results are skipped on a rerun; use `--force` to recompute. Each experiment's JSON and a status file are written below `work/ids2018_reduced_ext/results/`. A failed experiment is named in `_run_status.json`; a partial report is not a reproduction of the complete study.

## 4. Run the comparison profiles used by the report

The five-oracle reference and the first frozen seven-oracle configuration share the prepared CSE-CIC-IDS2018 corpus. The CICIDS2017 application excludes the transfer rule so no labels from its evaluation target enter training.

```bash
python -m twws.cli run-all --config configs/paper_reduced.yaml
python -m twws.cli run --config configs/paper_reduced_ext_frozen.yaml search main
python -m twws.cli pairs --config configs/ids2017_ext.yaml
python -m twws.cli label-check --config configs/ids2017_ext.yaml
python -m twws.cli run --config configs/ids2017_ext.yaml main
python -m twws.cli run --config configs/ids2017_ext_frozen.yaml main
```

The frozen CICIDS2017 run reads hyperparameters from the frozen CSE-CIC-IDS2018 search. `cross_dataset` is part of the main `run-all` sequence.

## 5. Generate a local report

```bash
python -m twws.cli report --config configs/paper_reduced_ext.yaml --out work/report
```

The report generator writes numbers, tables, figures, CSVs, and a SHA-256 manifest of the result files it read. These are generated locally under `work/report`; no precomputed outputs are included here. Some supplementary historical diagnostics were not retained, so their corresponding report cells may be absent unless you rerun the diagnostic commands in `twws.experiments.diagnostics` and pass their output with `--set report.diag_dir=DIR`.

Seeds are fixed in the configurations. Numerical results can vary slightly with library versions, hardware, and thread count. The load test's latency depends on the host machine.

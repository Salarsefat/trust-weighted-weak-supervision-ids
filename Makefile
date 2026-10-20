PY ?= python3
# the configuration of the article; the corpus is built with paper_reduced.yaml (see docs/REPRODUCE.md)
CONFIG ?= configs/paper_reduced_ext.yaml
CORPUS_CONFIG ?= configs/paper_reduced.yaml
OUT ?= work/report

.PHONY: test smoke list ingest corpus run report clean-smoke
test:
	$(PY) -m pytest -q
smoke:
	$(PY) -m twws.cli synthetic --config configs/smoke.yaml
	$(PY) -m twws.cli run-all --config configs/smoke.yaml
	$(PY) -m twws.cli report --config configs/smoke.yaml --out work/smoke/generated
list:
	$(PY) -m twws.cli fetch --config $(CORPUS_CONFIG) --list-only
ingest:
	$(PY) -m twws.cli ingest --config $(CORPUS_CONFIG) --delete-pcaps
corpus:
	$(PY) -m twws.cli corpus --config $(CORPUS_CONFIG)
	$(PY) -m twws.cli transfer --config $(CORPUS_CONFIG)
	$(PY) -m twws.cli pairs --config $(CORPUS_CONFIG)
run:
	$(PY) -m twws.cli label-check --config $(CONFIG)
	$(PY) -m twws.cli run-all --config $(CONFIG)
report:
	$(PY) -m twws.cli report --config $(CONFIG) --out $(OUT)
clean-smoke:
	rm -rf work/smoke work/smoke_src

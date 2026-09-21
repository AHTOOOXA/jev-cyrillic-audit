export HF_HOME ?= $(CURDIR)/hf_cache
.PHONY: reproduce reproduce1 reproduce2 reproduce3 test freeze check-prereg data run run2 run3

reproduce: reproduce1 reproduce2 reproduce3

reproduce1:
	uv run python -m jev_cyrillic_audit.analyse --runs runs --out results.json --figures figures

reproduce2:
	uv run python -m jev_cyrillic_audit.panorama --runs runs/study2 --out results2.json --figures figures

reproduce3:
	uv run python -m jev_cyrillic_audit.mechanism --runs runs/study3 --out results3.json --figures figures

test:
	uv run pytest -q

freeze:
	shasum -a 256 PREREG*.md > PREREG.sha256
	cd prompts && shasum -a 256 *.json > prompts.sha256

check-prereg:
	shasum -a 256 -c PREREG.sha256
	cd prompts && shasum -a 256 -c prompts.sha256

data:
	uv run python -m jev_cyrillic_audit.data --out data/items.parquet

run:
	uv run python -m jev_cyrillic_audit.run --run-id $(shell date -u +%Y%m%dT%H%M%SZ)

run2:
	uv run python -m jev_cyrillic_audit.run --study 2 --run-id $(shell date -u +%Y%m%dT%H%M%SZ)

run3:
	uv run python -m jev_cyrillic_audit.run --study 3 --run-id $(shell date -u +%Y%m%dT%H%M%SZ)

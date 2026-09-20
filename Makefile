export HF_HOME ?= $(CURDIR)/hf_cache
.PHONY: reproduce test freeze check-prereg data run

reproduce:
	uv run python -m jev_cyrillic_audit.analyse --runs runs --out results.json --figures figures

test:
	uv run pytest -q

freeze:
	shasum -a 256 PREREG.md > PREREG.sha256
	cd prompts && shasum -a 256 *.json > prompts.sha256

check-prereg:
	shasum -a 256 -c PREREG.sha256
	cd prompts && shasum -a 256 -c prompts.sha256

data:
	uv run python -m jev_cyrillic_audit.data --out data/items.parquet

run:
	uv run python -m jev_cyrillic_audit.run --run-id $(shell date -u +%Y%m%dT%H%M%SZ)

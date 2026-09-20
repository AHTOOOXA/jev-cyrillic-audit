# jev-cyrillic-audit

> Verdict: **not yet measured.** This line is replaced by one sentence after the first
> pre-registered run.

An independent, reproducible measurement of whether TypeSafe's Jev (`jev-1.13.0`) keeps its
**accuracy and calibration** on Russian vs English, on the *same human-labelled items*.

- Data: MASSIVE intent (`mteb/amazon_massive_intent`, en/ru) and XNLI (`facebook/xnli`, en/ru),
  n=600 per cell, two passes. Derived rows only are published — never source text.
- Metrics: accuracy, macro-F1, ECE (10 bins) with its simulated noise floor, paired RU−EN deltas
  with 1,000-resample bootstrap CIs, selective accuracy vs coverage, a determinism table, and the
  RU/EN input-token ratio.
- Protocol: `PREREG.md` and `prompts/` are frozen (SHA-256 committed) before the run; the model
  id is asserted on every response; `make reproduce` regenerates every number from `runs/*.jsonl`.

## Reproduce

```
uv sync
make reproduce        # runs/*.jsonl -> results.json -> figures/
```

## Layout

```
PREREG.md, PREREG.sha256      frozen before the run
prompts/*.json, prompts.sha256 instruction + criteria per dataset × language
data/items.parquet            item_id, dataset, lang, gold, join key — no text
runs/*.jsonl, manifest.json   one object per call, raw
src/jev_cyrillic_audit/       client, questions, metrics, analyse
tests/                        metrics unit tests on synthetic data
figures/                      reliability_paired.png, selective_accuracy.png
```

## Credits

Metric definitions and the pinned-revision discipline follow
[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) (Apache-2.0); any code
vendored from it keeps its notice in `THIRD_PARTY.md`.

## License

MIT for the code in this repository. Dataset licenses are the datasets' own; see `PREREG.md`.

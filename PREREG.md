# Pre-registration — jev-cyrillic-audit

> Status: **DRAFT. Not frozen.** Freeze = fill every field, `shasum -a 256 PREREG.md > PREREG.sha256`,
> commit both, and only then run anything that is not a dry run. Edits after freezing go in
> a dated "Deviations" section at the bottom, never in the body.

## Research question
Does `jev-1.13.0` keep (a) accuracy and (b) calibration on Russian inputs relative to English,
on the same items?

## Model
- `jev-1.13.0` (pinned; every response asserted; run aborts on drift)
- `typesafe-sdk==0.7.0`; endpoint `https://api.typesafe.ai`

## Data
| Dataset | HF id | Config / split | n per lang | Revision (commit hash) |
|---|---|---|---|---|
| MASSIVE intent | `mteb/amazon_massive_intent` | `en`, `ru` / test | 600 | TODO |
| XNLI | `facebook/xnli` | `en`, `ru` / test | 600 | TODO |

Sampling: `seed=20260919`, stratified by gold label, same item ids in both languages.

## Design
- Cells: state language ∈ {EN, RU}; instruction language = EN (MVP). One item per call.
- Option keys English in every cell. Prompts frozen in `prompts/` with `prompts.sha256`.
- Two passes per cell for the determinism table.
- Concurrency 8, client-side ≤ 1,150 rpm.

## Metrics (all defined in `src/jev_cyrillic_audit/metrics.py`)
accuracy · macro-F1 · ECE, 10 equal-width bins, on `p_max` · simulated ECE noise floor at the
same n and bins (1,000 sims) and the ratio · reliability diagram · selective accuracy vs
coverage + AURC · paired RU−EN deltas, 1,000-resample bootstrap 95% CI · label flip rate and
probability spread across passes · RU/EN input-token ratio.

## Decision rule (written before the run)
TODO — e.g. "Δacc CI excludes 0 and |Δacc| ≥ 3pp → 'measurably worse on RU';
Δece ratio > 2× floor → 'less calibrated on RU'; otherwise 'no detectable difference at n=600'".

## Not in scope for this run
Belebele, SIB-200, DaNetQA, toxicity, RU instructions, negation probes, other models.

## Deviations
(none)

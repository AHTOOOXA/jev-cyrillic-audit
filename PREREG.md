# Pre-registration — jev-cyrillic-audit

> Status: **DRAFT — to be frozen before the first non-dry run.** Freeze = `make freeze` (writes
> `PREREG.sha256` and `prompts/prompts.sha256`), commit both, and only then run anything that is
> not a `--scratch` dry run. The runner refuses to write to `runs/` unless both hashes verify and
> the frozen files are clean in git. Edits after freezing go in the dated "Deviations" section at
> the bottom, never in the body.

## Research question
Does `jev-1.13.0` keep (a) accuracy and (b) calibration on Russian inputs relative to English,
on the same human-labelled items?

The claim under test is the vendor's own (docs.typesafe.ai/introduction/machine-learning-primer):
*"Outcomes assigned a probability of 0.8 should occur about 80% of the time. Outcomes assigned a
probability of 1.0 should occur 100% of the time."* — together with the language caveat on
docs.typesafe.ai/models: *"English is the primary training language and where accuracy is
currently best. Other languages … are handled but not equally well."*

## Model and access
- `jev-1.13.0` (pinned; `response.model` asserted on every call; the run aborts on drift)
- `typesafe-sdk==0.7.0`, Python 3.11, endpoint default (`console.typesafe.ai` key)
- Price assumed for cost accounting: $0.042 per 1M input tokens, output free

## Data
| Dataset | HF id | Config / split | Pool | n per lang | Revision (commit) |
|---|---|---|---|---|---|
| XNLI | `facebook/xnli` | `en`, `ru` / `test` | 5,008 of 5,010 | 600 | `b8dd5d7af51114dbda02c0e3f6133f332186418e` |
| MASSIVE intent | `mteb/amazon_massive_intent` | `en`, `ru` / `test` | 2,974 | 600 | `940fd47a81eaa7f2cc7b129674d945d618ac38c2` |

- **Parallelism asserted before sampling:** XNLI is parallel by row index; MASSIVE is joined on
  `id`. Gold must be identical across languages on every item.
- **XNLI rows 2805 and 2806** have their two hypotheses swapped within the premise triple in the
  `ru` file (each language's label matches its own text; the row-index join is broken for exactly
  these two). They are excluded from the pool; the loader asserts the misaligned set is exactly
  `{2805, 2806}` at the pinned revision.
- **Sampling:** `seed=20260919`, proportional stratification by gold label (largest-remainder
  allocation), uniform without replacement within label, same item ids in both languages.
  XNLI: 200 / 200 / 200. MASSIVE: 58 of the 59 intents present in the test split
  (`general_greet`, n=1 in the split, receives no slot). Frozen in `data/items.parquet`
  (ids, gold, SHA-256 of each language's state — no text); its SHA-256 goes in `manifest.json`.
- **State passed to the model:** XNLI `{"premise": …, "hypothesis": …}`; MASSIVE
  `{"utterance": …}`. Dict keys are English in every cell.

## Design
- **Cells run in this study:** A = EN state, B = RU state; instruction language = EN in both.
  (Cells C/D, RU instructions, are frozen in `prompts/*.ru.json` but **not run** in this study.)
- One item per call, one `Choice` question per call — a deliberate departure from the vendor's
  "batch many questions per request" guidance, so that each item is judged without other items
  in the state (vendor failure mode #5, context rot) and per-item token counts are clean.
- Instructions reference the state fields by backticked key, as the vendor docs prescribe for
  object states (``Given `premise`, is `hypothesis` true, undetermined, or false?``;
  ``Which intent does the user's `utterance` express?``).
- Option keys are English identifiers in every cell (`entailment`, `alarm_set`, …). MASSIVE
  options carry a one-line **English** gloss each, identical in both arms (vendor docs: `null`
  descriptions are for self-explanatory names; several MASSIVE intents are not). English keys and
  glosses in the RU arm are a known leak and are listed as a limitation.
- Prompts frozen in `prompts/` with `prompts.sha256`. One wording per dataset — a sample of size 1
  from wording space; stated as a limitation.
- **Two passes** per cell (`pass ∈ {0, 1}`), pass 1 run after pass 0 has completed for all cells.
  **The headline uses pass 0.** Pass 1 is the stability check and is never averaged into the headline.
- **Identical-request check (dry run, before freeze).** The vendor's consistency cookbooks add a
  throwaway `uid` field to every call and state they "cannot separate sensitivity to the irrelevant
  field from variation that would occur on identical requests"; whether the API caches identical
  requests is undocumented. The `--scratch` dry run sends 20 items twice, byte-identical. If any
  answer differs, passes are sent identical (`uid` off). If all 40 are bit-identical and pass-1
  latency is markedly lower, pass 1 is sent with a `uid` field in the state and this is recorded
  here before the freeze. Result: **TODO (fill from the dry run).**
- Concurrency 8, client-side pacer ≤ 1,150 rpm, SDK retries honour `retry-after-ms`.
- Per-row log: `item_id, dataset, lang, instr_lang, pass, gold, pred, choice, p_max, confidence,
  probs, input_tokens, latency_ms, model, request_id, ts`. **Never the source text.**
- Abort rules: any `response.model != "jev-1.13.0"`; projected run cost > $1.
- Failed calls (API error after retries) are logged to `*.errors.jsonl`, retried on resume; an
  item still missing in either language after the run is dropped from **both** arms of the
  paired analysis and the count is reported.

## Metrics (all in `src/jev_cyrillic_audit/metrics.py`, numpy only, unit-tested on synthetic data)
- **Predicted label** = `argmax(probabilities)`; `p_max` = its probability. Rows where the API's
  `choice` differs from the argmax are counted and reported (argmax is used).
- Accuracy; macro-F1 over labels present in gold.
- **ECE**: 10 equal-width bins over `[0, 1]` on `p_max`, top bin closed, sample-weighted
  |accuracy − mean confidence|. Cross-checked once against `netcal` `ECE(bins=10)` to 1e-9.
- **ECE noise floor**: parametric bootstrap under perfect calibration on the arm's own `p_max`
  vector — labels drawn correct with probability `p_max`, 1,000 simulations, seed 0; report
  ECE, floor, and **ratio = ECE / floor** for every arm.
- Reliability diagram (bin means, bin counts printed), both languages on one panel per dataset.
- Selective accuracy vs coverage and AURC, computed on `confidence` (what the vendor says to gate
  on) and on `p_max`; anchors `(1−a) + a·ln a` (optimal) and `1−a` (uninformative).
- **CIs**: percentile bootstrap over items, 1,000 resamples, seed 0. **Paired** RU−EN deltas
  resample item indices once per replicate and recompute both arms on them. Marginal CIs are
  reported but never compared to each other.
- Determinism table per arm: label flip rate between passes, Cohen's κ between passes,
  mean and max |Δp_max|, and the flip-rate difference RU−EN.
- RU/EN input-token ratio per dataset from `usage.input_tokens` (same items, same prompt).
- `confidence` vs `p_max`: Pearson r and the share of rows with `confidence > p_max`, per arm.
- Accuracy and share of items in the `p_max == 1.0` bucket, per arm (the vendor's "1.0 → 100%").
- Coverage and accuracy at the vendor's canonical gates `confidence ≥ 0.5` and `≥ 0.9`, per arm.

## Headline statistics (named before the run)
Per dataset, on pass 0, EN instructions:
1. **Primary:** paired Δacc = acc_RU − acc_EN with its 95% paired bootstrap CI.
2. **Co-primary:** paired ΔECE = ECE_RU − ECE_EN on `p_max` with its 95% paired bootstrap CI,
   reported next to both arms' floors and ratios.

XNLI and MASSIVE are two pre-specified comparisons; they are not pooled. Everything else in the
metrics list is exploratory and labelled so in the README.

## Decision rule (written before the run)
Applied per dataset, in this order:

1. **Stability gate.** If, in either arm, |acc(pass 0) − acc(pass 1)| ≥ |Δacc(RU−EN)|, the
   accuracy verdict is **AMBIGUOUS** ("smaller than run-to-run jitter"). An arm with a label
   flip rate > 10% is additionally flagged **unstable** in the README.
2. **Accuracy.** If the Δacc CI excludes 0 and |Δacc| ≥ 3 pp → "measurably worse/better on
   Russian". If the CI includes 0 and its half-width ≤ 3 pp → "no detectable accuracy difference
   at n=600". Otherwise → **AMBIGUOUS at n=600**.
3. **Calibration measurability.** If ECE/floor < 1.5 in *either* arm → "this sample cannot
   measure calibration for <dataset>"; the calibration verdict is **AMBIGUOUS** and ΔECE is
   reported without a verdict.
4. **Calibration.** Otherwise: ΔECE CI excludes 0 and |ΔECE| ≥ 0.02 → "less/more calibrated on
   Russian"; CI includes 0 → "no detectable calibration difference at n=600"; else
   **AMBIGUOUS at n=600**.

The README's first sentence states both verdicts for both datasets, with the numbers.

## Sample size
n = 600 per cell gives ±2.5 pp on the paired accuracy delta at 10% RU/EN discordance and an ECE
floor near 0.03–0.05 at 10 bins, enough to separate a ratio-2 effect. Larger n is a follow-up,
not a deviation.

## Not in scope for this run
Belebele, SIB-200, DaNetQA, TERRa, toxicity, RU instructions (cells C/D), negation probes,
baseline models, any calibration fix (temperature scaling etc.), second prompt wordings.

## Licenses and redistribution
Derived rows only (`runs/*.jsonl`, `data/items.parquet`); no source text is redistributed.
MASSIVE: `apache-2.0` on the mirror, CC BY 4.0 upstream (attributed). XNLI: no license tag on
the HF card (UNVERIFIED); only our own measurements are published.

## Deviations
(none)

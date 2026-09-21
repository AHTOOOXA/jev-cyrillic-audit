# Pre-registration — Study 2, "Panorama": does tokenization cost predict Jev's cross-lingual calibration loss?

> Status: **FROZEN 2026-09-21 before the first non-dry run of Study 2.** Listed in `PREREG.sha256`
> next to `PREREG.md` (Study 1, unchanged); verify with `make check-prereg`. The runner refuses to
> write to `runs/study2/` unless the hashes verify and the frozen files are clean in git. Edits after
> this point go in "Deviations".

## Motivation (from Study 1, 2026-09-20)
Study 1 found a large Russian-vs-English loss on XNLI (Δacc −11.0 pp, ΔECE +0.063, both CIs
excluding 0) and no detectable loss on MASSIVE (Δacc −1.5 pp, ΔECE +0.005). "Russian" alone
therefore does not explain the degradation. Study 1 also measured that Russian costs 3.1–3.4× the
input tokens of English for the same text. Study 2 asks whether **tokenization cost** — a
label-free quantity anyone can compute on their own data — predicts the calibration loss across
languages, and whether the XNLI-vs-MASSIVE split is about task length rather than language.

## Research questions
- **RQ1 (primary).** Across the 14 non-English XNLI languages, is the calibration loss relative to
  English, ΔECE(L) = ECE_L − ECE_EN on the same items, positively associated with the language's
  state-token ratio r(L) = tokens_L / tokens_EN?
- **RQ2 (secondary).** Is the accuracy loss Δacc(L) negatively associated with r(L)?
- **RQ3 (exploratory).** Does the "hedging" seen in Russian (excess `neutral` predictions)
  generalise: is the excess-neutral share associated with Δacc(L)?
- **RQ4 (exploratory, task-length control).** On Belebele (reading comprehension, long passage,
  4-way MC), is the Russian loss closer to XNLI's (long input) or to MASSIVE's (short input)?

## Model and access
As Study 1: `jev-1.13.0` pinned and asserted; `typesafe-sdk==0.7.0`; $0.042 per 1M input tokens.

## Data
| Dataset | HF id | Configs | Split | Items | Revision |
|---|---|---|---|---|---|
| XNLI | `facebook/xnli` | `ar bg de el en es fr hi ru sw th tr ur vi zh` (all 15) | `test` | **the same 600 item ids frozen in Study 1** (`data/items.parquet`) | `b8dd5d7af51114dbda02c0e3f6133f332186418e` |
| Belebele | `facebook/belebele` | `eng_Latn`, `rus_Cyrl` | `test` | all 900, joined on `link` + `question_number` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` |

- XNLI is parallel by row index across all 15 configs. Per language, rows whose label differs from
  the `en` row are misaligned and are dropped for that language only (Study 1 found `ru` rows
  2805/2806); each language is paired with English on the remaining items. **Measured before the
  run:** misaligned rows exist only in `ru` ({2805, 2806}); the other 13 non-English configs have
  none, and the two `ru` rows are already outside the frozen 600 — so every language has all 600
  items and the item set is identical across all 15 arms (`data/items2.meta.json`). Script checks
  (Cyrillic, Arabic, Devanagari, Thai, Greek, CJK) pass at ≥ 0.997 for every non-Latin config.
- Belebele: `gold` = `correct_answer_num` mapped to `A–D`; asserted identical across en/ru per item.
- Frozen in `data/items2.parquet` (long format: dataset, item_id, lang, gold, state hash, criteria
  hash — no text).

## Design
- One `Choice` per call, one item per call, **English instructions in every cell**, English option
  keys. XNLI uses the Study-1 prompt `prompts/xnli.en.json` unchanged for all 15 languages.
  Belebele uses `prompts/belebele.en.json`: state `{"passage", "question"}`, criteria = the item's
  four answer texts in the item's language under fixed keys `A–D`.
- **All 15 XNLI languages are run in this study, including `en` and `ru`** (the reference arm and
  every comparison arm come from the same session; the Study-1 `en`/`ru` rows are not reused and
  serve as a cross-day stability check).
- Two passes per cell, identical requests (Study 1 established there is no response cache);
  **headline = pass 0**, pass 1 = stability.
- Cells: 15 XNLI + 2 Belebele = 17; 21,600 calls. **Dry run `dry2` (2026-09-21, 80 calls, 10 items ×
  {en, ru, hi, zh, ar, th, belebele-en, belebele-ru}, gitignored `scratch/study2/`):** 0 errors,
  p50 ≈ 300 ms, projected 11.7M input tokens ≈ **$0.49**, ≈ 19 min. Non-English answers were read by
  eye only to confirm they are sensible; prompts were not modified after the dry run.
- Concurrency 8, pacer ≤ 1,150 rpm, abort on model drift, budget guard $1.

## Measures
- Per arm (as Study 1): accuracy, macro-F1, ECE (10 bins on `p_max`, same binning), ECE noise floor
  and ratio, reliability bins, gates, determinism.
- **r(L)** = state-only token ratio = Σ(input_tokens_L − Q) / Σ(input_tokens_EN − Q) over the paired
  items, where Q is the per-dataset question overhead measured once with a one-character state
  (`data/question_overhead.json`: XNLI = 390; Belebele = 354, measured 2026-09-21 with one-character state and one-character answers). Bootstrap
  CI over items.
- **ΔECE(L), Δacc(L)**: paired deltas vs English with 1,000-resample paired bootstrap CIs.
- **Excess-neutral share(L)** = share of `neutral` predictions in L minus that in EN.

## Primary test (RQ1) — written before the run
- Statistic: **Spearman rank correlation ρ between r(L) and ΔECE(L) over the 14 non-English
  languages.** Rank-based because the token ratios are heavy-tailed (Thai/Hindi/Arabic ≫ German).
- Significance: one-sided permutation test (10,000 permutations of the language labels), H1: ρ > 0.
- Uncertainty: bootstrap over items — resample item indices 1,000× (same indices for all 15 arms,
  over the items present in every arm; measured before the run: all 600), recompute every ΔECE(L)
  and r(L), recompute ρ; report the 2.5–97.5 % interval.
- **Decision rule.** *Supported* if permutation p < 0.05 **and** the bootstrap interval of ρ excludes 0.
  *Not supported* if the point estimate ρ ≤ 0.2 and p ≥ 0.2. Otherwise **AMBIGUOUS at 14 languages**.
  With n = 14 languages the test has low power; the interval is the result even when the verdict is
  AMBIGUOUS.
- Pre-declared competing covariate (exploratory, not a test): script (Latin vs non-Latin). Reported
  alongside; not used to rescue H1.

RQ2 uses the same machinery with Δacc(L) and H2: ρ < 0. RQ3/RQ4 are descriptive.

## Chart
`figures/panorama_ece_vs_tokens.png`: x = r(L) (log scale), y = ΔECE(L) with paired 95 % CI bars,
one labelled point per language, English at (1, 0); title carries ρ with its interval and p.
Belebele ru plotted as a distinct marker. A companion panel with Δacc(L).

## Not in scope
Other tasks, other models, RU instructions, recalibration, negation probes.

## Deviations
(none)

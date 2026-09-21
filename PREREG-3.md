# Pre-registration — Study 3, "Mechanism": is Jev's cross-lingual loss a retreat to `neutral`, and does it happen when there is no `neutral`?

> Status: **FROZEN 2026-09-21 before the first non-dry run of Study 3.** Listed in `PREREG.sha256`
> next to `PREREG.md` and `PREREG-2.md` (both unchanged). The runner refuses to write to `runs/study3/`
> unless the hashes verify and the frozen files are clean in git. Edits after this point go in "Deviations".

## Motivation
Study 2 (2026-09-21) refuted its pre-registered hypothesis (tokenization cost does not predict
calibration loss) and produced an **exploratory** finding on the same 600 XNLI items: across 14
languages the accuracy loss vs English tracks the excess share of `neutral` predictions almost
perfectly (Spearman ρ = −0.95; 85.5 % of the items lost in a non-English language were predicted
`neutral`; the median recall gap vs English is 0.21 on `entailment`, 0.11 on `contradiction`,
≈ 0 on `neutral`). Exploratory findings on the items they were noticed on are not results. Study 3
tests the mechanism on **fresh items**, asks whether the loss survives when the "undetermined"
option is removed, and checks that a selection task does not degrade across the same languages.

## Data (frozen in `data/items3.parquet`; ids, gold, hashes — no text)
| Dataset id | Source | Items | Languages |
|---|---|---|---|
| `xnli_fresh` | `facebook/xnli` **validation** split (2,490; aligned across all 15 configs — measured, zero misaligned rows), revision `b8dd5d7a…`, stratified 200/200/200, `seed=20260921`; none of the items appears in Studies 1–2 | 600 | all 15 |
| `xnli_bin_ent` | the same 600 items, gold collapsed to `entailment` vs `not_entailment` | 600 | en, ru, sw, de, th, bg |
| `xnli_bin_con` | the same 600 items, gold collapsed to `contradiction` vs `not_contradiction` | 600 | en, ru |
| `sib200` | `Davlan/sib200` test (revision `38977a667f6fc264d5c26ec57a01e16db040b358`, CC BY-SA 4.0), 7 topics, parallel by `index_id`, categories asserted identical across languages | 204 | all 15 |

## Design
- Same protocol as Studies 1–2: `jev-1.13.0` asserted per call; one item, one `Choice` per call;
  English instructions and English option keys everywhere; two passes, identical requests; headline
  = pass 0; concurrency 8 under a 1,150 rpm pacer; abort on drift; budget guard $1.
- Prompts: `xnli_fresh` uses the Study-1 prompt `prompts/xnli.en.json` **unchanged**;
  `prompts/xnli_bin_ent.en.json` ("Given `premise`, is `hypothesis` true?" → `entailment` /
  `not_entailment`), `prompts/xnli_bin_con.en.json` ("… is `hypothesis` false?" → `contradiction` /
  `not_contradiction`), `prompts/sib200.en.json` (7 topics with one-line English glosses). All hashed
  in `prompts/prompts.sha256`; question overheads measured with one-character states
  (`data/question_overhead.json`: 390 / 368 / 369 / 427).
- 38 cells, 33,720 calls; dry run `dry3` (80 calls, gitignored `scratch/study3/`): 0 errors,
  projected ≈ 16.8M tokens ≈ **$0.71**, ≈ 30 min. Non-English answers read by eye only.

## Hypotheses, tests and decision rules (written before the run)

**H1 — confirmatory replication of the neutral-drift mechanism** (`xnli_fresh`, 14 languages vs `en`, pass 0).
- (a) Spearman ρ between excess-`neutral` share (L − EN) and Δacc (L − EN) across the 14 languages
  < 0: one-sided permutation p (10,000 permutations) < 0.05 **and** the item-bootstrap 95 % interval
  of ρ (1,000 resamples, same indices for all arms) lies below 0. *Prediction from Study 2: ρ ≈ −0.95.*
- (b) Pooled share of lost items (EN correct, L wrong) predicted `neutral` ≥ 0.70. *Prediction: ≈ 0.85.*
  (If errors were spread over the two wrong classes this share would be ≈ 0.5.)
- **Decision:** CONFIRMED if (a) and (b) both hold. NOT CONFIRMED if ρ ≥ −0.3 or the share < 0.55.
  Otherwise AMBIGUOUS.

**H2 — hedging vs recognition failure** (`xnli_bin_ent` vs `xnli_fresh`, L ∈ {ru, sw, de, th, bg}, gold-`entailment` items, n = 200 per language).
- R3(L) = recall_ent(EN, 3-way) − recall_ent(L, 3-way); R2(L) = recall_ent(EN, binary) − recall_ent(L, binary);
  recovery(L) = 1 − R2(L)/R3(L). A language with R3(L) < 0.02 is excluded from the mean and reported.
- Statistic: mean recovery over the included languages, with an item-bootstrap 95 % interval.
- **Decision:** mean recovery ≥ 0.5 → "hedging dominant" (removing the undetermined option recovers
  most of the entailment loss); ≤ 0.2 → "recognition failure dominant" (the model does not see the
  entailment in that language whatever the framing); otherwise "mixed".
- Guard against trivial recovery (saying `entailment` to everything): report per language the binary
  task's recall on `not_entailment` items and the paired Δ balanced accuracy vs EN; recovery counts
  only if the language's binary balanced accuracy is within 5 pp of its 3-way-derived value
  (recall_ent and 1 − FPR computed from the same items). `xnli_bin_con` (ru, en) is the same
  analysis for `contradiction`, exploratory.

**H3 — selection tasks do not degrade across languages** (`sib200`, 14 languages vs `en`).
- Statistic: median over the 14 languages of paired Δacc(L − EN); each language also gets its CI.
- **Decision:** supported if the median Δacc(SIB-200) ≥ −3 pp **and** median Δacc(`xnli_fresh`) −
  median Δacc(SIB-200) ≤ −5 pp (the inference task loses at least 5 pp more than the selection task
  on the same languages); not supported if median Δacc(SIB-200) ≤ −6 pp; otherwise AMBIGUOUS.
  n = 204 per language: per-language CIs are wide (±5 pp) and are reported, not tested.

Also reported (exploratory): ΔECE per language and task; token ratios; determinism per arm;
whether ρ(token ratio, ΔECE) on the fresh items stays ≈ 0 (Study 2's negative result).

## Charts
`figures/mechanism_fresh_drift.png` (H1 line on fresh items, Study-2 line as a faint reference),
`figures/mechanism_recovery.png` (H2 recall bars per language, 3-way vs binary, EN vs L),
`figures/mechanism_tasks.png` (H3: Δacc per language, XNLI-fresh vs SIB-200).

## Not in scope
Other models, RU instructions, recalibration, negation probes, MASSIVE/Belebele in more languages.

## Deviations
(none)

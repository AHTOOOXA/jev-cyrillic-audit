# jev-cyrillic-audit

> **Verdict (jev-1.13.0, n=600 paired items per dataset, pre-registered):** on XNLI, Jev is **measurably worse and less calibrated in Russian** — accuracy 88.3% → 77.3% (paired Δ = -11.0 pp, 95% CI [-14.2, -7.8]) and ECE 0.032 → 0.096 (Δ = +0.063 [+0.033, +0.088]); on MASSIVE intent classification there is **no detectable difference at n=600** — accuracy 86.7% vs 85.2% (Δ = -1.5 pp [-3.5, +0.2]), ECE 0.066 vs 0.071 (Δ = +0.005 [-0.011, +0.026]).

![Paired reliability diagram, English vs Russian, XNLI and MASSIVE](figures/reliability_paired.png)

An independent, reproducible measurement of whether TypeSafe's Jev (`jev-1.13.0`) keeps its
**accuracy and calibration** on Russian vs English, on the *same human-labelled items*. The vendor
says non-English is "handled but not equally well" and gives no number
([docs](https://docs.typesafe.ai/models#language-support)); its own calibration claim is that
"outcomes assigned a probability of 0.8 should occur about 80% of the time"
([primer](https://docs.typesafe.ai/introduction/machine-learning-primer)). This repo tests exactly
that, in Russian, with the noise floor of the metric reported next to the metric.

Every number below regenerates from the committed raw responses with `make reproduce`.

## Results (headline pass, English instructions, English option keys)

## XNLI  (n=600 paired items, pass 0, jev-1.13.0)
| metric | EN | RU | RU − EN (paired 95% CI) |
|---|---|---|---|
| accuracy | 0.883 [0.858, 0.910] | 0.773 [0.743, 0.805] | **-11.0 pp** [-14.2, -7.8] |
| macro-F1 | 0.885 | 0.776 | -10.9 pp [-13.9, -7.7] |
| ECE (10 bins, p_max) | 0.032 [0.019, 0.056] | 0.096 [0.069, 0.126] | **+0.063** [+0.033, +0.088] |
| ECE noise floor / ratio | 0.018 / 1.74× | 0.024 / 4.06× | |
| mean p_max | 0.915 | 0.869 | |
| share p_max = 1.0 → accuracy there | 41.0% → 0.992 | 15.2% → 0.945 | |
| confidence ≥ 0.5: coverage → accuracy | 89.5% → 0.927 | 86.2% → 0.818 | |
| confidence ≥ 0.9: coverage → accuracy | 70.5% → 0.972 | 51.8% → 0.887 | |
| AURC on confidence (optimal / random) | 0.0279 (0.0071 / 0.1167) | 0.1094 (0.0279 / 0.2267) | |
| input tokens / call | 423 | 491 | ratio 1.162; state-only **3.07×** [3.01, 3.14] |
| discordance (exactly one arm correct) | | | 17.3%; same prediction in both arms 82.3% |

**Determinism (pass 0 vs pass 1, identical requests)**

| arm | flip rate | κ | mean \|Δp_max\| | max \|Δp_max\| | identical prob vectors | acc pass0 / pass1 |
|---|---|---|---|---|---|---|
| en | 1.50% | 0.977 | 0.0088 | 0.14 | 61.8% | 0.883 / 0.885 |
| ru | 1.67% | 0.973 | 0.0132 | 0.10 | 42.0% | 0.773 / 0.772 |

**Verdict (xnli):** accuracy — measurably worse on Russian; calibration — less calibrated on Russian.
confidence vs p_max: r=1.000/1.000, conf>p_max 0.0%/0.0%; choice≠argmax 2/0

## MASSIVE  (n=600 paired items, pass 0, jev-1.13.0)
| metric | EN | RU | RU − EN (paired 95% CI) |
|---|---|---|---|
| accuracy | 0.867 [0.840, 0.893] | 0.852 [0.823, 0.877] | **-1.5 pp** [-3.5, +0.2] |
| macro-F1 | 0.883 | 0.869 | -1.4 pp [-3.2, +2.0] |
| ECE (10 bins, p_max) | 0.066 [0.047, 0.091] | 0.071 [0.055, 0.099] | **+0.005** [-0.011, +0.026] |
| ECE noise floor / ratio | 0.018 / 3.73× | 0.020 / 3.61× | |
| mean p_max | 0.929 | 0.918 | |
| share p_max = 1.0 → accuracy there | 59.2% → 0.972 | 55.7% → 0.973 | |
| confidence ≥ 0.5: coverage → accuracy | 96.2% → 0.889 | 95.7% → 0.868 | |
| confidence ≥ 0.9: coverage → accuracy | 80.3% → 0.944 | 77.0% → 0.935 | |
| AURC on confidence (optimal / random) | 0.0325 (0.0093 / 0.1333) | 0.0426 (0.0116 / 0.1483) | |
| input tokens / call | 1460 | 1476 | ratio 1.010; state-only **3.37×** [3.26, 3.48] |
| discordance (exactly one arm correct) | | | 5.2%; same prediction in both arms 92.5% |

**Determinism (pass 0 vs pass 1, identical requests)**

| arm | flip rate | κ | mean \|Δp_max\| | max \|Δp_max\| | identical prob vectors | acc pass0 / pass1 |
|---|---|---|---|---|---|---|
| en | 0.50% | 0.995 | 0.0081 | 0.14 | 70.0% | 0.867 / 0.865 |
| ru | 2.17% | 0.978 | 0.0086 | 0.14 | 62.5% | 0.852 / 0.855 |

**Verdict (massive):** accuracy — no detectable accuracy difference at n=600; calibration — no detectable calibration difference at n=600.
confidence vs p_max: r=1.000/1.000, conf>p_max 0.2%/0.2%; choice≠argmax 0/1

Reading the XNLI row of the reliability diagram: in the top bin (`p_max ≥ 0.9`, 360 Russian items) Jev states
0.97 on average and is right 88.6% of the time; in English the same bin states 0.985 and is right 96.4%.
The Russian accuracy loss is concentrated on `entailment` (recall 0.85 → 0.64): in Russian the model
drifts toward `neutral` (303 vs 236 predictions), i.e. it hedges more and is still overconfident when it does.

![Selective accuracy vs coverage, English vs Russian](figures/selective_accuracy.png)

**Other measurements (exploratory, pre-registered as such):**

- **Gating on the vendor's example thresholds.** XNLI at `confidence ≥ 0.9`: English acts on 70% of items at 97.2% accuracy, Russian on 52% at 88.7%. MASSIVE: 80% → 94.4% vs 77% → 93.5%.
- **Russian costs ~3× the tokens for the same text.** State-only input tokens RU/EN: **3.07×** (XNLI) and **3.37×** (MASSIVE). Per call the ratio is diluted by the fixed question text (1.16× / 1.01×). Latency is identical (p50 ≈ 320–340 ms).
- **Run-to-run stability is high.** Identical requests flip the label on 0.5–2.2% of items (κ ≥ 0.97), |Δp_max| ≤ 0.14; accuracy moves ≤ 0.3 pp between passes, so the RU−EN deltas are not jitter. The API does **not** cache identical requests (probability vectors differ on 30–58% of repeated calls).
- **`confidence` is a function of `p_max`.** Across all 4,800 responses `confidence` matches `(k·p_max − 1)/(k − 1)` (k = number of options) to within the API's 2-decimal rounding (max residual 0.02, r = 1.000). Calibration is therefore computed on `p_max`, which is comparable with the literature; ranking by `confidence` and by `p_max` is the same ranking within a dataset.
- **Ties.** 7 of 4,800 responses have `choice` ≠ argmax of the returned probabilities — all are exact 2-decimal ties (0.50/0.50). The API breaks ties on unrounded values; we use the argmax and report the count.
- **Probabilities are returned rounded to 2 decimals.** Bin edges are therefore stated explicitly (left-closed, exact decimal edges, top bin closed); ECE is cross-checked against `netcal` to 1e-9 in the tests.

## Method

- **Design.** Two factors would make a 2×2 (state language × instruction language); this run is the clean pair **A = English state vs B = Russian state, English instructions in both**, on the same items. One item per call, one `Choice` question per call, option keys English in both arms (`entailment`, `alarm_set`, …). Two passes per cell; the headline uses pass 0, pass 1 is the stability check.
- **Data** (pinned HF revisions in `PREREG.md`): XNLI `facebook/xnli` en/ru test (5,010 human-translated pairs; 2 rows with swapped hypotheses in the `ru` file excluded before sampling) and MASSIVE intent `mteb/amazon_massive_intent` en/ru test (2,974 utterances, 59 intents). 600 items per dataset, stratified by gold label, `seed=20260919`, frozen in `data/items.parquet` (ids and hashes only — **no source text is redistributed**).
- **Prompts** (`prompts/*.json`, SHA-256 frozen before the run) follow the vendor's docs: state as a JSON object, fields referenced by backticked key in the instructions, a one-line English gloss per MASSIVE intent.
- **Protocol.** `PREREG.md` + hashes committed before the first non-dry run; `response.model` asserted on every call; concurrency 8 under a 1,150 rpm pacer; every response logged as a derived row (`item_id, gold, pred, p_max, confidence, probs, input_tokens, latency_ms, model, request_id, ts`). 4,800 calls, 0 errors, 4.6M input tokens, **$0.19**, 4 min 15 s.
- **Metrics** (`src/jev_cyrillic_audit/metrics.py`, numpy only, 11 synthetic tests): accuracy, macro-F1, ECE (10 equal-width bins on `p_max`) with its **simulated noise floor** (1,000 draws under perfect calibration on the arm's own confidence vector) and the ratio, reliability bins, selective accuracy / AURC, percentile bootstrap CIs (1,000 resamples), and **paired** RU−EN deltas that resample item indices once and recompute both arms — never "the two CIs overlap".
- **Decision rule** (written before the run, applied in order): stability gate (run-to-run jitter must be smaller than the delta) → accuracy: CI excludes 0 and |Δ| ≥ 3 pp → "measurably worse/better"; CI includes 0 with half-width ≤ 3 pp → "no detectable difference"; else AMBIGUOUS → calibration measurable only if ECE/floor ≥ 1.5 in both arms → ΔECE: CI excludes 0 and |Δ| ≥ 0.02 → "less/more calibrated"; CI includes 0 → "no detectable difference"; else AMBIGUOUS.

## Limitations

- One prompt wording per dataset (a sample of size 1 from wording space). Option keys and MASSIVE glosses are English in the Russian arm — a deliberate control that also means the Russian cell is not a fully-Russian deployment. The RU-instruction cells (C/D) are frozen in `prompts/*.ru.json` but not run here.
- XNLI labels were annotated on the English text and copied to the translations; ~15% of XNLI items are ambiguous even to humans. That ceiling applies to both arms equally, which is why the comparison is paired.
- n=600 resolves ±3 pp on the paired accuracy delta and a ratio-2 calibration effect; the MASSIVE result is "not detected at this n", not "equal". ECE bins below 0.9 hold 1–84 items (counts printed under the chart).
- Black-box API: results are for `jev-1.13.0` on 2026-09-20; the raw responses are committed so the numbers survive model updates even if the API does not.
- Two datasets, one model, no baseline system. Baselines (fine-tuned `xlm-roberta`, a zero-shot LLM with log-probs) are follow-up work, not part of this measurement.

## Reproduce

```
uv sync
make check-prereg      # PREREG.md and prompts/ match their committed hashes
make test              # 19 tests: metrics on synthetic data (incl. netcal cross-check), runner on a fake API
make reproduce         # runs/*.jsonl -> results.json + results.md + figures/*.png (deterministic)
```

To re-run the measurement itself (needs `TYPESAFE_API_KEY` in `.env`; ~$0.20):
`uv run python -m jev_cyrillic_audit.run --run-id <new-id>`. The runner refuses to write to `runs/`
unless the freeze verifies and the frozen files are clean in git. Building a new sample:
`uv run python -m jev_cyrillic_audit.data`.

## Layout

```
PREREG.md, PREREG.sha256       frozen before the run; decision rule inside
prompts/*.json, prompts.sha256 instructions + criteria per dataset × instruction language
data/items.parquet             item ids, gold, state hashes — no text; data/question_overhead.json
runs/<run_id>-*.jsonl          raw derived rows, one per call; runs/manifest.json
results.json, results.md       every number in this README, regenerated by `make reproduce`
figures/                       reliability_paired.png, selective_accuracy.png
src/jev_cyrillic_audit/        data.py · questions.py · run.py · metrics.py · analyse.py · figures.py
tests/                         synthetic metric tests, fake-API runner tests
```

## Credits and license

Method conventions (pinned revisions, per-example artefacts, paired bootstrap) follow
[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) (Apache-2.0); see `THIRD_PARTY.md`.
ECE noise-floor framing after Guo et al. 2017 and the `netcal` reference implementation.

MIT for the code and the derived rows in this repository. Datasets keep their own licenses
(MASSIVE: Apache-2.0 on the mirror / CC BY 4.0 upstream; XNLI: no license tag on the HF card) — only
our own measurements are redistributed, re-join on `item_id` to recover the text.

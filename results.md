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

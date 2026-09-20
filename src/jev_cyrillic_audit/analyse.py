"""runs/*.jsonl -> results.json (+ figures via `figures.py`). Every number in the README comes from here.

    python -m jev_cyrillic_audit.analyse --runs runs --out results.json --figures figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .run import MODEL

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("xnli", "massive")
LANGS = ("en", "ru")
GATES = (0.5, 0.9)
MIN_ACC_DELTA_PP = 3.0     # decision rule §2
MIN_ECE_RATIO = 1.5        # decision rule §3
MIN_ECE_DELTA = 0.02       # decision rule §4
UNSTABLE_FLIP_RATE = 0.10  # decision rule §1


def load_runs(runs_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((runs_dir / "manifest.json").read_text())
    files = [runs_dir / c["file"] for c in manifest["cells"].values()]
    df = pd.DataFrame([json.loads(l) for f in files for l in f.open(encoding="utf-8")])
    assert (df["model"] == MODEL).all(), df["model"].unique()
    df["correct"] = df["pred"] == df["gold"]
    return df, manifest


def arm(df: pd.DataFrame, dataset: str, lang: str, p: int) -> pd.DataFrame:
    """One cell at one pass, indexed by item_id, in a fixed item order."""
    a = df[(df.dataset == dataset) & (df.lang == lang) & (df["pass"] == p)].set_index("item_id").sort_index()
    assert not a.index.duplicated().any()
    return a


def paired(df: pd.DataFrame, dataset: str, p: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    en, ru = arm(df, dataset, "en", p), arm(df, dataset, "ru", p)
    common = en.index.intersection(ru.index)
    dropped = len(en.index.union(ru.index)) - len(common)
    return en.loc[common], ru.loc[common], dropped


def arm_metrics(a: pd.DataFrame) -> dict:
    c, pmax, conf = a["correct"].to_numpy(float), a["p_max"].to_numpy(float), a["confidence"].to_numpy(float)
    acc = M.accuracy(a["gold"], a["pred"])
    e, floor = M.ece(c, pmax), M.ece_floor(pmax)
    bins = M.reliability_bins(c, pmax)
    top = pmax >= 1.0
    return {
        "n": int(len(a)),
        "accuracy": acc, "accuracy_ci": M.boot_ci(M.accuracy, a["gold"].to_numpy(), a["pred"].to_numpy()),
        "macro_f1": M.macro_f1(a["gold"], a["pred"]),
        "ece": e, "ece_ci": M.boot_ci(M.ece, c, pmax), "ece_floor": floor, "ece_ratio": e / floor if floor else None,
        "reliability": {k: v.tolist() for k, v in bins.items()},
        "aurc_confidence": M.aurc(c, conf), "aurc_pmax": M.aurc(c, pmax), "aurc_anchors": M.aurc_anchors(acc),
        "gates": {str(g): M.coverage_at_gate(c, conf, g) for g in GATES},
        "pmax_1_share": float(top.mean()), "pmax_1_accuracy": float(c[top].mean()) if top.any() else None,
        "mean_pmax": float(pmax.mean()), "mean_confidence": float(conf.mean()),
        "confidence_vs_pmax": {"pearson_r": float(np.corrcoef(conf, pmax)[0, 1]),
                               "share_conf_gt_pmax": float(np.mean(conf > pmax + 1e-12)),
                               "share_conf_eq_pmax": float(np.mean(np.abs(conf - pmax) < 1e-12))},
        "choice_ne_argmax": int((a["choice"] != a["pred"]).sum()),
        "input_tokens_mean": float(a["input_tokens"].mean()),
        "latency_p50_ms": float(a["latency_ms"].median()),
    }


def determinism_metrics(a0: pd.DataFrame, a1: pd.DataFrame) -> dict:
    common = a0.index.intersection(a1.index)
    a0, a1 = a0.loc[common], a1.loc[common]
    d = M.determinism(a0["pred"].to_numpy(), a1["pred"].to_numpy(), a0["p_max"].to_numpy(), a1["p_max"].to_numpy())
    d["n"] = int(len(common))
    d["accuracy_pass0"] = M.accuracy(a0["gold"], a0["pred"])
    d["accuracy_pass1"] = M.accuracy(a1["gold"], a1["pred"])
    d["abs_acc_diff_between_passes"] = abs(d["accuracy_pass0"] - d["accuracy_pass1"])
    d["share_identical_prob_vectors"] = float(np.mean([a0.loc[i, "probs"] == a1.loc[i, "probs"] for i in common]))
    return d


def token_ratio(en: pd.DataFrame, ru: pd.DataFrame, overhead: int) -> dict:
    te, tr = en["input_tokens"].to_numpy(float), ru["input_tokens"].to_numpy(float)
    se, sr = te - overhead, tr - overhead
    ratio = lambda a, b: float(a.sum() / b.sum())
    return {
        "question_overhead_tokens": overhead,
        "per_call_ratio": ratio(tr, te), "per_call_ratio_ci": M.boot_ci(lambda a, b: a.sum() / b.sum(), tr, te),
        "state_only_ratio": ratio(sr, se), "state_only_ratio_ci": M.boot_ci(lambda a, b: a.sum() / b.sum(), sr, se),
        "mean_state_tokens_en": float(se.mean()), "mean_state_tokens_ru": float(sr.mean()),
    }


def decide(r: dict) -> dict:
    """PREREG decision rule, applied per dataset in the written order."""
    en, ru, d = r["en"], r["ru"], r["paired"]
    acc_delta_pp = d["accuracy"]["delta"] * 100
    lo, hi = d["accuracy"]["ci_low"] * 100, d["accuracy"]["ci_high"] * 100
    half = (hi - lo) / 2
    jitter = max(r["determinism"]["en"]["abs_acc_diff_between_passes"], r["determinism"]["ru"]["abs_acc_diff_between_passes"]) * 100
    notes = []
    if jitter >= abs(acc_delta_pp):
        acc_v = "AMBIGUOUS"; notes.append(f"stability gate: run-to-run jitter {jitter:.1f} pp ≥ |Δacc| {abs(acc_delta_pp):.1f} pp")
    elif (lo > 0 or hi < 0) and abs(acc_delta_pp) >= MIN_ACC_DELTA_PP:
        acc_v = "measurably worse on Russian" if acc_delta_pp < 0 else "measurably better on Russian"
    elif lo <= 0 <= hi and half <= MIN_ACC_DELTA_PP:
        acc_v = "no detectable accuracy difference at n=600"
    else:
        acc_v = "AMBIGUOUS at n=600"
    for lang in LANGS:
        if r["determinism"][lang]["flip_rate"] > UNSTABLE_FLIP_RATE:
            notes.append(f"{lang} arm flagged unstable (flip rate {r['determinism'][lang]['flip_rate']:.1%})")
    if min(en["ece_ratio"], ru["ece_ratio"]) < MIN_ECE_RATIO:
        cal_v = "AMBIGUOUS"; notes.append(f"calibration not measurable: ECE/floor en {en['ece_ratio']:.2f}, ru {ru['ece_ratio']:.2f} (< {MIN_ECE_RATIO})")
    else:
        e = d["ece"]
        if (e["ci_low"] > 0 or e["ci_high"] < 0) and abs(e["delta"]) >= MIN_ECE_DELTA:
            cal_v = "less calibrated on Russian" if e["delta"] > 0 else "more calibrated on Russian"
        elif e["ci_low"] <= 0 <= e["ci_high"]:
            cal_v = "no detectable calibration difference at n=600"
        else:
            cal_v = "AMBIGUOUS at n=600"
    return {"accuracy": acc_v, "calibration": cal_v, "notes": notes}


def analyse(runs_dir: Path) -> dict:
    df, manifest = load_runs(runs_dir)
    overhead = json.loads((ROOT / "data/question_overhead.json").read_text())
    out = {"run_id": manifest["run_id"], "model": MODEL, "n_rows": int(len(df)), "headline_pass": 0,
           "manifest": {k: manifest[k] for k in ("sdk_version", "seed", "n_per_cell", "passes", "total_calls",
                                                  "total_input_tokens", "total_cost_usd", "run_started_utc",
                                                  "run_finished_utc", "prereg_sha256", "dataset_revisions")},
           "datasets": {}}
    for ds in DATASETS:
        en, ru, dropped = paired(df, ds, 0)
        c_en, c_ru = en["correct"].to_numpy(float), ru["correct"].to_numpy(float)
        p_en, p_ru = en["p_max"].to_numpy(float), ru["p_max"].to_numpy(float)
        r = {
            "n_paired": int(len(en)), "items_dropped": dropped,
            "en": arm_metrics(en), "ru": arm_metrics(ru),
            "paired": {
                "accuracy": M.paired_delta(M.accuracy, [ru["gold"].to_numpy(), ru["pred"].to_numpy()], [en["gold"].to_numpy(), en["pred"].to_numpy()]),
                "macro_f1": M.paired_delta(M.macro_f1, [ru["gold"].to_numpy(), ru["pred"].to_numpy()], [en["gold"].to_numpy(), en["pred"].to_numpy()]),
                "ece": M.paired_delta(M.ece, [c_ru, p_ru], [c_en, p_en]),
                "discordance": float(np.mean(c_en != c_ru)),
                "both_correct": float(np.mean((c_en == 1) & (c_ru == 1))),
                "same_prediction": float(np.mean(en["pred"].to_numpy() == ru["pred"].to_numpy())),
            },
            "determinism": {lang: determinism_metrics(arm(df, ds, lang, 0), arm(df, ds, lang, 1)) for lang in LANGS},
            "tokens": token_ratio(en, ru, overhead[ds]["input_tokens_with_minimal_state"]),
            "pass1_exploratory": {lang: {"accuracy": M.accuracy(arm(df, ds, lang, 1)["gold"], arm(df, ds, lang, 1)["pred"]),
                                         "ece": M.ece(arm(df, ds, lang, 1)["correct"].to_numpy(float), arm(df, ds, lang, 1)["p_max"].to_numpy(float))}
                                  for lang in LANGS},
        }
        r["determinism"]["flip_rate_ru_minus_en"] = r["determinism"]["ru"]["flip_rate"] - r["determinism"]["en"]["flip_rate"]
        r["verdict"] = decide(r)
        out["datasets"][ds] = r
    return out


def fmt_table(res: dict) -> str:
    L = []
    for ds, r in res["datasets"].items():
        en, ru, d = r["en"], r["ru"], r["paired"]
        pp = lambda x: f"{x*100:+.1f}"
        L.append(f"\n## {ds.upper()}  (n={r['n_paired']} paired items, pass 0, jev-1.13.0)")
        L.append("| metric | EN | RU | RU − EN (paired 95% CI) |")
        L.append("|---|---|---|---|")
        L.append(f"| accuracy | {en['accuracy']:.3f} [{en['accuracy_ci'][0]:.3f}, {en['accuracy_ci'][1]:.3f}] | {ru['accuracy']:.3f} [{ru['accuracy_ci'][0]:.3f}, {ru['accuracy_ci'][1]:.3f}] | **{pp(d['accuracy']['delta'])} pp** [{pp(d['accuracy']['ci_low'])}, {pp(d['accuracy']['ci_high'])}] |")
        L.append(f"| macro-F1 | {en['macro_f1']:.3f} | {ru['macro_f1']:.3f} | {pp(d['macro_f1']['delta'])} pp [{pp(d['macro_f1']['ci_low'])}, {pp(d['macro_f1']['ci_high'])}] |")
        L.append(f"| ECE (10 bins, p_max) | {en['ece']:.3f} [{en['ece_ci'][0]:.3f}, {en['ece_ci'][1]:.3f}] | {ru['ece']:.3f} [{ru['ece_ci'][0]:.3f}, {ru['ece_ci'][1]:.3f}] | **{d['ece']['delta']:+.3f}** [{d['ece']['ci_low']:+.3f}, {d['ece']['ci_high']:+.3f}] |")
        L.append(f"| ECE noise floor / ratio | {en['ece_floor']:.3f} / {en['ece_ratio']:.2f}× | {ru['ece_floor']:.3f} / {ru['ece_ratio']:.2f}× | |")
        L.append(f"| mean p_max | {en['mean_pmax']:.3f} | {ru['mean_pmax']:.3f} | |")
        L.append(f"| share p_max = 1.0 → accuracy there | {en['pmax_1_share']:.1%} → {en['pmax_1_accuracy']:.3f} | {ru['pmax_1_share']:.1%} → {ru['pmax_1_accuracy']:.3f} | |")
        for g in GATES:
            ge, gr = en["gates"][str(g)], ru["gates"][str(g)]
            L.append(f"| confidence ≥ {g}: coverage → accuracy | {ge['coverage']:.1%} → {ge['accuracy']:.3f} | {gr['coverage']:.1%} → {gr['accuracy']:.3f} | |")
        L.append(f"| AURC on confidence (optimal / random) | {en['aurc_confidence']:.4f} ({en['aurc_anchors']['optimal']:.4f} / {en['aurc_anchors']['random']:.4f}) | {ru['aurc_confidence']:.4f} ({ru['aurc_anchors']['optimal']:.4f} / {ru['aurc_anchors']['random']:.4f}) | |")
        L.append(f"| input tokens / call | {en['input_tokens_mean']:.0f} | {ru['input_tokens_mean']:.0f} | ratio {r['tokens']['per_call_ratio']:.3f}; state-only **{r['tokens']['state_only_ratio']:.2f}×** [{r['tokens']['state_only_ratio_ci'][0]:.2f}, {r['tokens']['state_only_ratio_ci'][1]:.2f}] |")
        L.append(f"| discordance (exactly one arm correct) | | | {d['discordance']:.1%}; same prediction in both arms {d['same_prediction']:.1%} |")
        L.append("\n**Determinism (pass 0 vs pass 1, identical requests)**\n")
        L.append("| arm | flip rate | κ | mean \\|Δp_max\\| | max \\|Δp_max\\| | identical prob vectors | acc pass0 / pass1 |")
        L.append("|---|---|---|---|---|---|---|")
        for lang in LANGS:
            t = r["determinism"][lang]
            L.append(f"| {lang} | {t['flip_rate']:.2%} | {t['kappa']:.3f} | {t['mean_abs_dpmax']:.4f} | {t['max_abs_dpmax']:.2f} | {t['share_identical_prob_vectors']:.1%} | {t['accuracy_pass0']:.3f} / {t['accuracy_pass1']:.3f} |")
        v = r["verdict"]
        L.append(f"\n**Verdict ({ds}):** accuracy — {v['accuracy']}; calibration — {v['calibration']}." + (" Notes: " + "; ".join(v["notes"]) if v["notes"] else ""))
        L.append(f"confidence vs p_max: r={en['confidence_vs_pmax']['pearson_r']:.3f}/{ru['confidence_vs_pmax']['pearson_r']:.3f}, conf>p_max {en['confidence_vs_pmax']['share_conf_gt_pmax']:.1%}/{ru['confidence_vs_pmax']['share_conf_gt_pmax']:.1%}; choice≠argmax {en['choice_ne_argmax']}/{ru['choice_ne_argmax']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results.json"); ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    res = analyse(Path(a.runs))
    Path(a.out).write_text(json.dumps(res, indent=2, default=float) + "\n")
    table = fmt_table(res)
    Path(a.out).with_suffix(".md").write_text(table.strip() + "\n")
    print(table)
    if a.figures:
        from .figures import make_figures
        make_figures(res, Path(a.figures))


if __name__ == "__main__":
    main()

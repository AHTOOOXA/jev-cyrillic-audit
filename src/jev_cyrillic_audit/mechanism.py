"""Study 3 analysis: runs/study3/*.jsonl -> results3.json / results3.md / figures/mechanism_*.png.

H1 neutral-drift replication on fresh items; H2 hedging vs recognition failure via the binary framing;
H3 selection-task control (SIB-200) across the same languages. Decision rules in PREREG-3.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .analyse import arm, arm_metrics, determinism_metrics, load_runs, token_ratio
from .panorama import LATIN, NON_EN, N_BOOT, N_PERM, SEED, perm_p_one_sided, spearman
from .run import XNLI_LANGS

ROOT = Path(__file__).resolve().parents[2]
BIN_LANGS = ("ru", "sw", "de", "th", "bg")


def _pair(df, ds, lang, ref="en", p=0):
    a, e = arm(df, ds, lang, p), arm(df, ds, ref, p)
    common = a.index.intersection(e.index)
    return a.loc[common], e.loc[common]


def recall(a: pd.DataFrame, cls: str) -> float:
    m = a["gold"].to_numpy() == cls
    return float(np.mean(a["pred"].to_numpy()[m] == cls))


# ---------- H1 ----------

def h1(df: pd.DataFrame, overhead: int) -> dict:
    en = arm(df, "xnli_fresh", "en", 0)
    langs, lost_total, lost_neutral = {}, 0, 0
    for l in NON_EN:
        a, e = _pair(df, "xnli_fresh", l)
        ca, pa, ce, pe = a["correct"].to_numpy(float), a["p_max"].to_numpy(float), e["correct"].to_numpy(float), e["p_max"].to_numpy(float)
        lost = (e["correct"] & ~a["correct"]).to_numpy()
        langs[l] = {
            "lang": l, "script": "latin" if l in LATIN else "non-latin", "n_paired": int(len(a)), "arm": arm_metrics(a),
            "d_acc": M.paired_delta(M.accuracy, [a["gold"].to_numpy(), a["pred"].to_numpy()], [e["gold"].to_numpy(), e["pred"].to_numpy()]),
            "d_ece": M.paired_delta(M.ece, [ca, pa], [ce, pe]),
            "excess_neutral": float(np.mean(a["pred"] == "neutral") - np.mean(e["pred"] == "neutral")),
            "n_lost": int(lost.sum()), "lost_to_neutral": float(np.mean(a["pred"].to_numpy()[lost] == "neutral")) if lost.any() else None,
            "recall": {c: recall(a, c) for c in ("entailment", "neutral", "contradiction")},
            "tokens": token_ratio(e, a, overhead),
            "determinism": determinism_metrics(arm(df, "xnli_fresh", l, 0), arm(df, "xnli_fresh", l, 1)),
        }
        lost_total += int(lost.sum()); lost_neutral += int((a["pred"].to_numpy()[lost] == "neutral").sum())
    x = np.array([langs[l]["excess_neutral"] for l in NON_EN]); y = np.array([langs[l]["d_acc"]["delta"] for l in NON_EN])
    rho, p = spearman(x, y), perm_p_one_sided(x, y, -1)
    # item bootstrap of rho (same indices for all 15 arms; all 600 items are present in every arm)
    arms = {l: arm(df, "xnli_fresh", l, 0).loc[en.index] for l in XNLI_LANGS}
    corr = {l: arms[l]["correct"].to_numpy(float) for l in XNLI_LANGS}
    neu = {l: (arms[l]["pred"].to_numpy() == "neutral").astype(float) for l in XNLI_LANGS}
    rng = np.random.default_rng(SEED); n = len(en); rhos = np.empty(N_BOOT)
    for i in range(N_BOOT):
        k = rng.integers(0, n, n)
        xs = [neu[l][k].mean() - neu["en"][k].mean() for l in NON_EN]; ys = [corr[l][k].mean() - corr["en"][k].mean() for l in NON_EN]
        rhos[i] = spearman(xs, ys)
    ci = [float(v) for v in np.quantile(rhos, [0.025, 0.975])]
    share = lost_neutral / lost_total
    a_ok = p < 0.05 and ci[1] < 0; b_ok = share >= 0.70
    verdict = "CONFIRMED" if (a_ok and b_ok) else ("NOT CONFIRMED" if (rho >= -0.3 or share < 0.55) else "AMBIGUOUS")
    r_tok = np.array([langs[l]["tokens"]["state_only_ratio"] for l in NON_EN]); d_ece = np.array([langs[l]["d_ece"]["delta"] for l in NON_EN])
    return {"languages": langs, "en": {"arm": arm_metrics(en), "recall": {c: recall(en, c) for c in ("entailment", "neutral", "contradiction")},
                                       "determinism": determinism_metrics(en, arm(df, "xnli_fresh", "en", 1))},
            "test": {"rho": rho, "perm_p_one_sided": p, "rho_ci_items": ci, "pooled_lost_to_neutral": share, "n_lost_total": lost_total,
                     "slope_pp_per_pp": float(np.polyfit(x * 100, y * 100, 1)[0]), "verdict": verdict},
            "token_ratio_vs_dece_rho": spearman(r_tok, d_ece)}


# ---------- H2 ----------

def h2(df: pd.DataFrame) -> dict:
    out, recs = {}, []
    en3, en2 = arm(df, "xnli_fresh", "en", 0), arm(df, "xnli_bin_ent", "en", 0)
    for l in BIN_LANGS:
        l3, l2 = arm(df, "xnli_fresh", l, 0).loc[en3.index], arm(df, "xnli_bin_ent", l, 0).loc[en2.index]
        r3 = recall(en3, "entailment") - recall(l3, "entailment")
        r2 = recall(en2, "entailment") - recall(l2, "entailment")
        fpr3 = lambda a: float(np.mean(a["pred"].to_numpy()[a["gold"].to_numpy() != "entailment"] == "entailment"))
        ba3 = lambda a: (recall(a, "entailment") + 1 - fpr3(a)) / 2
        ba2 = lambda a: (recall(a, "entailment") + recall(a, "not_entailment")) / 2
        rec = None if r3 < 0.02 else 1 - r2 / r3
        out[l] = {"recall_ent_3way": {"en": recall(en3, "entailment"), "lang": recall(l3, "entailment")},
                  "recall_ent_binary": {"en": recall(en2, "entailment"), "lang": recall(l2, "entailment")},
                  "recall_notent_binary": {"en": recall(en2, "not_entailment"), "lang": recall(l2, "not_entailment")},
                  "gap_3way": r3, "gap_binary": r2, "recovery": rec,
                  "balanced_acc_3way_derived": ba3(l3), "balanced_acc_binary": ba2(l2),
                  "d_balanced_acc_binary_vs_en": ba2(l2) - ba2(en2),
                  "guard_ok": bool(abs(ba2(l2) - ba3(l3)) <= 0.05),
                  "binary_arm": arm_metrics(l2), "binary_en_arm": arm_metrics(en2),
                  "binary_determinism": determinism_metrics(arm(df, "xnli_bin_ent", l, 0), arm(df, "xnli_bin_ent", l, 1))}
        if rec is not None:
            recs.append(rec)
    # item bootstrap of the mean recovery: resample the 600 items once for all arms
    rng = np.random.default_rng(SEED); n = len(en3); vals = []
    A = {l: (arm(df, "xnli_fresh", l, 0).loc[en3.index], arm(df, "xnli_bin_ent", l, 0).loc[en2.index]) for l in BIN_LANGS}
    g3, g2 = en3["gold"].to_numpy(), en2["gold"].to_numpy()
    pe3, pe2 = en3["pred"].to_numpy(), en2["pred"].to_numpy()
    for i in range(N_BOOT):
        k = rng.integers(0, n, n); m3, m2 = g3[k] == "entailment", g2[k] == "entailment"
        r_en3, r_en2 = np.mean(pe3[k][m3] == "entailment"), np.mean(pe2[k][m2] == "entailment")
        rs = []
        for l, (l3, l2) in A.items():
            R3 = r_en3 - np.mean(l3["pred"].to_numpy()[k][m3] == "entailment")
            R2 = r_en2 - np.mean(l2["pred"].to_numpy()[k][m2] == "entailment")
            if R3 >= 0.02:
                rs.append(1 - R2 / R3)
        vals.append(np.mean(rs) if rs else np.nan)
    vals = np.array(vals); vals = vals[~np.isnan(vals)]
    mean_rec = float(np.mean(recs)) if recs else None
    ci = [float(v) for v in np.quantile(vals, [0.025, 0.975])] if len(vals) else None
    verdict = None if mean_rec is None else ("hedging dominant" if mean_rec >= 0.5 else "recognition failure dominant" if mean_rec <= 0.2 else "mixed")
    # exploratory: contradiction framing, ru
    enc, ruc = arm(df, "xnli_bin_con", "en", 0), arm(df, "xnli_bin_con", "ru", 0)
    ru3 = arm(df, "xnli_fresh", "ru", 0).loc[en3.index]
    con = {"recall_con_3way": {"en": recall(en3, "contradiction"), "ru": recall(ru3, "contradiction")},
           "recall_con_binary": {"en": recall(enc, "contradiction"), "ru": recall(ruc.loc[enc.index], "contradiction")},
           "recall_notcon_binary": {"en": recall(enc, "not_contradiction"), "ru": recall(ruc.loc[enc.index], "not_contradiction")}}
    g3c = con["recall_con_3way"]["en"] - con["recall_con_3way"]["ru"]; g2c = con["recall_con_binary"]["en"] - con["recall_con_binary"]["ru"]
    con["gap_3way"], con["gap_binary"], con["recovery"] = g3c, g2c, (None if g3c < 0.02 else 1 - g2c / g3c)
    return {"languages": out, "mean_recovery": mean_rec, "mean_recovery_ci_items": ci, "languages_included": [l for l in BIN_LANGS if out[l]["recovery"] is not None],
            "all_guards_ok": all(out[l]["guard_ok"] for l in BIN_LANGS), "verdict": verdict, "contradiction_ru_exploratory": con}


# ---------- H3 ----------

def h3(df: pd.DataFrame, h1res: dict, overhead: int) -> dict:
    langs = {}
    for l in NON_EN:
        a, e = _pair(df, "sib200", l)
        langs[l] = {"n_paired": int(len(a)), "arm": arm_metrics(a),
                    "d_acc": M.paired_delta(M.accuracy, [a["gold"].to_numpy(), a["pred"].to_numpy()], [e["gold"].to_numpy(), e["pred"].to_numpy()]),
                    "d_ece": M.paired_delta(M.ece, [a["correct"].to_numpy(float), a["p_max"].to_numpy(float)], [e["correct"].to_numpy(float), e["p_max"].to_numpy(float)]),
                    "tokens": token_ratio(e, a, overhead),
                    "determinism": determinism_metrics(arm(df, "sib200", l, 0), arm(df, "sib200", l, 1))}
    med_sib = float(np.median([langs[l]["d_acc"]["delta"] for l in NON_EN]))
    med_xnli = float(np.median([h1res["languages"][l]["d_acc"]["delta"] for l in NON_EN]))
    ok = med_sib >= -0.03 and (med_xnli - med_sib) <= -0.05
    verdict = "SUPPORTED" if ok else ("NOT SUPPORTED" if med_sib <= -0.06 else "AMBIGUOUS")
    en = arm(df, "sib200", "en", 0)
    return {"languages": langs, "en": {"arm": arm_metrics(en)}, "median_d_acc_sib200": med_sib, "median_d_acc_xnli_fresh": med_xnli,
            "n_langs_ci_excludes_0": int(sum(langs[l]["d_acc"]["ci_high"] < 0 for l in NON_EN)), "verdict": verdict}


def analyse3(runs_dir: Path) -> dict:
    df, manifest = load_runs(runs_dir)
    ov = json.loads((ROOT / "data/question_overhead.json").read_text())
    r1 = h1(df, ov["xnli"]["input_tokens_with_minimal_state"])
    r2 = h2(df)
    r3 = h3(df, r1, ov["sib200"]["input_tokens_with_minimal_state"])
    return {"run_id": manifest["run_id"], "study": 3, "model": manifest["model"], "n_rows": int(len(df)),
            "manifest": {k: manifest[k] for k in ("total_calls", "total_input_tokens", "total_cost_usd", "run_started_utc", "run_finished_utc", "prereg_sha256")},
            "h1_neutral_drift": r1, "h2_binary_framing": r2, "h3_sib200": r3}


def fmt_table(res: dict) -> str:
    L = []; pp = lambda v: f"{v*100:+.1f}"
    r1, r2, r3 = res["h1_neutral_drift"], res["h2_binary_framing"], res["h3_sib200"]
    t = r1["test"]
    L.append(f"## Study 3 — Mechanism (fresh XNLI validation items, n=600; pass 0; {res['model']})\n")
    L.append(f"English reference (fresh items): accuracy {r1['en']['arm']['accuracy']:.3f}, ECE {r1['en']['arm']['ece']:.3f}; recall ent/neu/con "
             f"{r1['en']['recall']['entailment']:.2f}/{r1['en']['recall']['neutral']:.2f}/{r1['en']['recall']['contradiction']:.2f}.\n")
    L.append("| lang | accuracy | Δacc vs EN (95% CI) | ΔECE vs EN (95% CI) | excess neutral | lost items → neutral | recall ent / con | flip rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for l in sorted(NON_EN, key=lambda x: r1["languages"][x]["d_acc"]["delta"]):
        r = r1["languages"][l]
        L.append(f"| {l} | {r['arm']['accuracy']:.3f} | {pp(r['d_acc']['delta'])} [{pp(r['d_acc']['ci_low'])}, {pp(r['d_acc']['ci_high'])}] | "
                 f"{r['d_ece']['delta']:+.3f} [{r['d_ece']['ci_low']:+.3f}, {r['d_ece']['ci_high']:+.3f}] | {pp(r['excess_neutral'])} pp | "
                 f"{r['lost_to_neutral']:.0%} of {r['n_lost']} | {r['recall']['entailment']:.2f} / {r['recall']['contradiction']:.2f} | {r['determinism']['flip_rate']:.1%} |")
    L.append(f"\n**H1 (confirmatory):** ρ(excess neutral, Δacc) = **{t['rho']:+.2f}**, permutation p = {t['perm_p_one_sided']:.4f}, item-bootstrap CI "
             f"[{t['rho_ci_items'][0]:+.2f}, {t['rho_ci_items'][1]:+.2f}]; pooled share of lost items predicted neutral = **{t['pooled_lost_to_neutral']:.1%}** "
             f"(n = {t['n_lost_total']}); slope {t['slope_pp_per_pp']:+.2f} pp/pp → **{t['verdict']}**. (ρ(token ratio, ΔECE) on fresh items = {r1['token_ratio_vs_dece_rho']:+.2f}.)")
    L.append(f"\n## H2 — remove the `neutral` option: recall on gold-entailment items (n=200), 3-way vs binary\n")
    L.append("| lang | recall_ent 3-way EN → L (gap) | recall_ent binary EN → L (gap) | recovery | recall not_ent binary EN / L | Δ balanced acc binary vs EN | guard |")
    L.append("|---|---|---|---|---|---|---|")
    for l in BIN_LANGS:
        r = r2["languages"][l]
        rec = "n/a" if r["recovery"] is None else f"{r['recovery']:+.0%}"
        L.append(f"| {l} | {r['recall_ent_3way']['en']:.2f} → {r['recall_ent_3way']['lang']:.2f} ({r['gap_3way']:+.2f}) | "
                 f"{r['recall_ent_binary']['en']:.2f} → {r['recall_ent_binary']['lang']:.2f} ({r['gap_binary']:+.2f}) | **{rec}** | "
                 f"{r['recall_notent_binary']['en']:.2f} / {r['recall_notent_binary']['lang']:.2f} | {pp(r['d_balanced_acc_binary_vs_en'])} pp | {'ok' if r['guard_ok'] else 'FAIL'} |")
    ci = r2["mean_recovery_ci_items"]
    L.append(f"\n**H2:** mean recovery = **{r2['mean_recovery']:+.0%}** [{ci[0]:+.0%}, {ci[1]:+.0%}] over {len(r2['languages_included'])} languages; guards {'all ok' if r2['all_guards_ok'] else 'FAILED'} → **{r2['verdict']}**")
    c = r2["contradiction_ru_exploratory"]
    rec_c = "n/a" if c["recovery"] is None else f"{c['recovery']:+.0%}"
    L.append(f"Exploratory, contradiction framing (ru): recall_con 3-way {c['recall_con_3way']['en']:.2f} → {c['recall_con_3way']['ru']:.2f}; "
             f"binary {c['recall_con_binary']['en']:.2f} → {c['recall_con_binary']['ru']:.2f}; recovery {rec_c}")
    L.append(f"\n## H3 — SIB-200 (7 topics, n=204 per language), the selection-task control\n")
    L.append(f"English reference: accuracy {r3['en']['arm']['accuracy']:.3f}, ECE {r3['en']['arm']['ece']:.3f}.\n")
    L.append("| lang | accuracy | Δacc vs EN (95% CI) | ΔECE vs EN | flip rate |\n|---|---|---|---|---|")
    for l in sorted(NON_EN, key=lambda x: r3["languages"][x]["d_acc"]["delta"]):
        r = r3["languages"][l]
        L.append(f"| {l} | {r['arm']['accuracy']:.3f} | {pp(r['d_acc']['delta'])} [{pp(r['d_acc']['ci_low'])}, {pp(r['d_acc']['ci_high'])}] | {r['d_ece']['delta']:+.3f} | {r['determinism']['flip_rate']:.1%} |")
    L.append(f"\n**H3:** median Δacc SIB-200 = **{pp(r3['median_d_acc_sib200'])} pp** vs median Δacc XNLI-fresh = {pp(r3['median_d_acc_xnli_fresh'])} pp; "
             f"{r3['n_langs_ci_excludes_0']}/14 SIB-200 languages have a CI below 0 → **{r3['verdict']}**")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/study3"); ap.add_argument("--out", default="results3.json"); ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    res = analyse3(Path(a.runs))
    Path(a.out).write_text(json.dumps(res, indent=2, default=float) + "\n")
    table = fmt_table(res); Path(a.out).with_suffix(".md").write_text(table.strip() + "\n"); print(table)
    if a.figures:
        from .figures import mechanism_charts
        mechanism_charts(res, Path(a.figures))


if __name__ == "__main__":
    main()

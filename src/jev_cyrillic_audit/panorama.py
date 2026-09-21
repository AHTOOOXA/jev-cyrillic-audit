"""Study 2 analysis: runs/study2/*.jsonl -> results2.json / results2.md / figures/panorama_ece_vs_tokens.png.

Primary test (PREREG-2): Spearman rho between the state-token ratio r(L) and the paired calibration
loss dECE(L) over the 14 non-English XNLI languages; one-sided permutation p; bootstrap over items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .analyse import arm, arm_metrics, determinism_metrics, load_runs, token_ratio
from .run import XNLI_LANGS

ROOT = Path(__file__).resolve().parents[2]
NON_EN = tuple(l for l in XNLI_LANGS if l != "en")
LATIN = {"de", "en", "es", "fr", "sw", "tr", "vi"}
N_PERM = 10_000
N_BOOT = 1000
SEED = 0


# ---------- rank statistics (numpy only) ----------

def rankdata(x) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, float)
    ranks[order] = np.arange(1, x.size + 1)
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p_one_sided(x, y, sign: int, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """P(rho_perm >= rho_obs) for sign=+1 (H1: rho > 0), or <= for sign=-1."""
    rng = np.random.default_rng(seed)
    obs = spearman(x, y) * sign
    y = np.asarray(y, float)
    hits = sum(spearman(x, rng.permutation(y)) * sign >= obs - 1e-12 for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


# ---------- per-language paired analysis ----------

def paired_lang(df: pd.DataFrame, lang: str, ref: str = "en", p: int = 0):
    a, e = arm(df, "xnli", lang, p), arm(df, "xnli", ref, p)
    common = a.index.intersection(e.index)
    return a.loc[common], e.loc[common]


def lang_row(df: pd.DataFrame, lang: str, overhead: int) -> dict:
    a, e = paired_lang(df, lang)
    ca, pa = a["correct"].to_numpy(float), a["p_max"].to_numpy(float)
    ce, pe = e["correct"].to_numpy(float), e["p_max"].to_numpy(float)
    neutral = lambda x: float(np.mean(x["pred"].to_numpy() == "neutral"))
    r = {
        "lang": lang, "script": "latin" if lang in LATIN else "non-latin", "n_paired": int(len(a)),
        "arm": arm_metrics(a),
        "d_acc": M.paired_delta(M.accuracy, [a["gold"].to_numpy(), a["pred"].to_numpy()], [e["gold"].to_numpy(), e["pred"].to_numpy()]),
        "d_ece": M.paired_delta(M.ece, [ca, pa], [ce, pe]),
        "tokens": token_ratio(e, a, overhead),
        "neutral_share": neutral(a), "excess_neutral": neutral(a) - neutral(e),
        "determinism": determinism_metrics(arm(df, "xnli", lang, 0), arm(df, "xnli", lang, 1)),
    }
    return r


def rho_bootstrap(df: pd.DataFrame, overhead: int, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Resample the items present in all 15 arms; recompute every dECE(L), r(L), and rho."""
    arms = {l: arm(df, "xnli", l, 0) for l in XNLI_LANGS}
    common = arms["en"].index
    for l in NON_EN:
        common = common.intersection(arms[l].index)
    A = {l: arms[l].loc[common] for l in XNLI_LANGS}
    corr = {l: A[l]["correct"].to_numpy(float) for l in XNLI_LANGS}
    pmax = {l: A[l]["p_max"].to_numpy(float) for l in XNLI_LANGS}
    tok = {l: A[l]["input_tokens"].to_numpy(float) - overhead for l in XNLI_LANGS}
    rng = np.random.default_rng(seed)
    n = len(common)
    rhos_e, rhos_a = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        ece_en, acc_en, t_en = M.ece(corr["en"][k], pmax["en"][k]), corr["en"][k].mean(), tok["en"][k].sum()
        d_e, d_a, r = [], [], []
        for l in NON_EN:
            d_e.append(M.ece(corr[l][k], pmax[l][k]) - ece_en)
            d_a.append(corr[l][k].mean() - acc_en)
            r.append(tok[l][k].sum() / t_en)
        rhos_e[i], rhos_a[i] = spearman(r, d_e), spearman(r, d_a)
    q = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]
    return {"n_common_items": int(n), "rho_ece_ci": q(rhos_e), "rho_acc_ci": q(rhos_a)}


def decide(rho: float, p: float, ci: list[float], sign: int) -> str:
    if p < 0.05 and (ci[0] > 0 if sign > 0 else ci[1] < 0):
        return "SUPPORTED"
    if (rho * sign) <= 0.2 and p >= 0.2:
        return "NOT SUPPORTED"
    return "AMBIGUOUS at 14 languages"


def analyse2(runs_dir: Path) -> dict:
    df, manifest = load_runs(runs_dir)
    overhead = json.loads((ROOT / "data/question_overhead.json").read_text())
    q_x = overhead["xnli"]["input_tokens_with_minimal_state"]
    q_b = overhead["belebele"]["input_tokens_with_minimal_state"]
    langs = {l: lang_row(df, l, q_x) for l in NON_EN}
    en = arm(df, "xnli", "en", 0)
    ref = {"arm": arm_metrics(en), "neutral_share": float(np.mean(en["pred"] == "neutral")),
           "determinism": determinism_metrics(en, arm(df, "xnli", "en", 1))}
    r = np.array([langs[l]["tokens"]["state_only_ratio"] for l in NON_EN])
    d_ece = np.array([langs[l]["d_ece"]["delta"] for l in NON_EN])
    d_acc = np.array([langs[l]["d_acc"]["delta"] for l in NON_EN])
    x_neu = np.array([langs[l]["excess_neutral"] for l in NON_EN])
    boot = rho_bootstrap(df, q_x)
    rho_e, rho_a = spearman(r, d_ece), spearman(r, d_acc)
    p_e, p_a = perm_p_one_sided(r, d_ece, +1), perm_p_one_sided(r, d_acc, -1)
    tests = {
        "rq1_ece_vs_tokens": {"rho": rho_e, "perm_p_one_sided": p_e, "rho_ci_items": boot["rho_ece_ci"],
                              "verdict": decide(rho_e, p_e, boot["rho_ece_ci"], +1)},
        "rq2_acc_vs_tokens": {"rho": rho_a, "perm_p_one_sided": p_a, "rho_ci_items": boot["rho_acc_ci"],
                              "verdict": decide(rho_a, p_a, boot["rho_acc_ci"], -1)},
        "rq3_excess_neutral_vs_dacc": {"rho": spearman(x_neu, d_acc), "pearson_r": float(np.corrcoef(x_neu, d_acc)[0, 1])},
        "exploratory_script": {
            "mean_d_ece_latin": float(np.mean([langs[l]["d_ece"]["delta"] for l in NON_EN if l in LATIN])),
            "mean_d_ece_non_latin": float(np.mean([langs[l]["d_ece"]["delta"] for l in NON_EN if l not in LATIN])),
            "mean_ratio_latin": float(np.mean([langs[l]["tokens"]["state_only_ratio"] for l in NON_EN if l in LATIN])),
            "mean_ratio_non_latin": float(np.mean([langs[l]["tokens"]["state_only_ratio"] for l in NON_EN if l not in LATIN])),
            "rho_within_non_latin": spearman([langs[l]["tokens"]["state_only_ratio"] for l in NON_EN if l not in LATIN],
                                             [langs[l]["d_ece"]["delta"] for l in NON_EN if l not in LATIN]),
        },
        "n_common_items": boot["n_common_items"],
    }
    # Belebele en vs ru, as Study 1
    be, br = arm(df, "belebele", "en", 0), arm(df, "belebele", "ru", 0)
    common = be.index.intersection(br.index); be, br = be.loc[common], br.loc[common]
    bel = {
        "n_paired": int(len(common)), "en": arm_metrics(be), "ru": arm_metrics(br),
        "d_acc": M.paired_delta(M.accuracy, [br["gold"].to_numpy(), br["pred"].to_numpy()], [be["gold"].to_numpy(), be["pred"].to_numpy()]),
        "d_ece": M.paired_delta(M.ece, [br["correct"].to_numpy(float), br["p_max"].to_numpy(float)], [be["correct"].to_numpy(float), be["p_max"].to_numpy(float)]),
        "tokens": token_ratio(be, br, q_b),
        "determinism": {l: determinism_metrics(arm(df, "belebele", l, 0), arm(df, "belebele", l, 1)) for l in ("en", "ru")},
    }
    return {"run_id": manifest["run_id"], "study": 2, "model": manifest["model"], "n_rows": int(len(df)),
            "manifest": {k: manifest[k] for k in ("total_calls", "total_input_tokens", "total_cost_usd", "run_started_utc", "run_finished_utc", "prereg_sha256")},
            "xnli_en_reference": ref, "xnli_languages": langs, "tests": tests, "belebele": bel}


def fmt_table(res: dict) -> str:
    L = []
    t = res["tests"]; en = res["xnli_en_reference"]["arm"]
    L.append(f"## Study 2 — Panorama (XNLI × 15 languages, same 600 items; pass 0; {res['model']})\n")
    L.append(f"English reference: accuracy {en['accuracy']:.3f}, ECE {en['ece']:.3f} (floor {en['ece_floor']:.3f}).\n")
    L.append("| lang | script | n | tokens ×EN | accuracy | Δacc vs EN (95% CI) | ECE (floor) | ΔECE vs EN (95% CI) | excess neutral | flip rate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = sorted(res["xnli_languages"].values(), key=lambda r: r["tokens"]["state_only_ratio"])
    pp = lambda v: f"{v*100:+.1f}"
    for r in rows:
        a = r["arm"]
        L.append(f"| {r['lang']} | {r['script']} | {r['n_paired']} | {r['tokens']['state_only_ratio']:.2f} | {a['accuracy']:.3f} | "
                 f"{pp(r['d_acc']['delta'])} [{pp(r['d_acc']['ci_low'])}, {pp(r['d_acc']['ci_high'])}] | {a['ece']:.3f} ({a['ece_floor']:.3f}) | "
                 f"{r['d_ece']['delta']:+.3f} [{r['d_ece']['ci_low']:+.3f}, {r['d_ece']['ci_high']:+.3f}] | {pp(r['excess_neutral'])} pp | {r['determinism']['flip_rate']:.1%} |")
    e, a_ = t["rq1_ece_vs_tokens"], t["rq2_acc_vs_tokens"]
    L.append(f"\n**RQ1 (primary):** Spearman ρ(token ratio, ΔECE) = **{e['rho']:+.2f}**, one-sided permutation p = {e['perm_p_one_sided']:.3f}, "
             f"item-bootstrap 95% CI [{e['rho_ci_items'][0]:+.2f}, {e['rho_ci_items'][1]:+.2f}] (n = {t['n_common_items']} common items) → **{e['verdict']}**")
    L.append(f"**RQ2:** ρ(token ratio, Δacc) = **{a_['rho']:+.2f}**, p = {a_['perm_p_one_sided']:.3f}, CI [{a_['rho_ci_items'][0]:+.2f}, {a_['rho_ci_items'][1]:+.2f}] → **{a_['verdict']}**")
    L.append(f"**RQ3:** ρ(excess neutral, Δacc) = {t['rq3_excess_neutral_vs_dacc']['rho']:+.2f} (Pearson {t['rq3_excess_neutral_vs_dacc']['pearson_r']:+.2f})")
    s = t["exploratory_script"]
    L.append(f"**Script (exploratory):** mean ΔECE Latin {s['mean_d_ece_latin']:+.3f} (ratio {s['mean_ratio_latin']:.2f}×) vs non-Latin {s['mean_d_ece_non_latin']:+.3f} (ratio {s['mean_ratio_non_latin']:.2f}×); ρ within non-Latin {s['rho_within_non_latin']:+.2f}")
    b = res["belebele"]
    L.append(f"\n## Belebele (reading comprehension, 4-way MC), EN vs RU, n={b['n_paired']} paired\n")
    L.append("| metric | EN | RU | RU − EN (paired 95% CI) |\n|---|---|---|---|")
    L.append(f"| accuracy | {b['en']['accuracy']:.3f} | {b['ru']['accuracy']:.3f} | **{pp(b['d_acc']['delta'])} pp** [{pp(b['d_acc']['ci_low'])}, {pp(b['d_acc']['ci_high'])}] |")
    L.append(f"| ECE (floor) | {b['en']['ece']:.3f} ({b['en']['ece_floor']:.3f}) | {b['ru']['ece']:.3f} ({b['ru']['ece_floor']:.3f}) | **{b['d_ece']['delta']:+.3f}** [{b['d_ece']['ci_low']:+.3f}, {b['d_ece']['ci_high']:+.3f}] |")
    L.append(f"| state tokens ×EN | | | {b['tokens']['state_only_ratio']:.2f} [{b['tokens']['state_only_ratio_ci'][0]:.2f}, {b['tokens']['state_only_ratio_ci'][1]:.2f}] |")
    L.append(f"| flip rate | {b['determinism']['en']['flip_rate']:.1%} | {b['determinism']['ru']['flip_rate']:.1%} | |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/study2"); ap.add_argument("--out", default="results2.json"); ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    res = analyse2(Path(a.runs))
    Path(a.out).write_text(json.dumps(res, indent=2, default=float) + "\n")
    table = fmt_table(res)
    Path(a.out).with_suffix(".md").write_text(table.strip() + "\n")
    print(table)
    if a.figures:
        from .figures import panorama_chart
        panorama_chart(res, Path(a.figures) / "panorama_ece_vs_tokens.png")


if __name__ == "__main__":
    main()

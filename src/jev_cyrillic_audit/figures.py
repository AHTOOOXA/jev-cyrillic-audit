"""The two charts. Colours: EN blue, RU orange (fixed slots, CVD-validated); text in ink tokens."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as M
from .analyse import GATES, ROOT, arm, load_runs

C = {"en": "#2a78d6", "ru": "#eb6834"}
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1"
NAME = {"en": "English", "ru": "Russian"}
TITLE = {"xnli": "XNLI (3-way NLI)", "massive": "MASSIVE (59 intents)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _fmt_delta(d: dict, scale=1.0, unit="", nd=3) -> str:
    f = lambda x: f"{x*scale:+.{nd}f}"
    return f"{f(d['delta'])}{unit} [{f(d['ci_low'])}, {f(d['ci_high'])}]"


def reliability_paired(res: dict, out: Path) -> None:
    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.24], hspace=0.08, wspace=0.28)
    for j, ds in enumerate(("xnli", "massive")):
        r = res["datasets"][ds]
        ax = fig.add_subplot(gs[0, j]); axc = fig.add_subplot(gs[1, j], sharex=ax)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK3, zorder=1)
        ax.text(0.13, 0.21, "underconfident", ha="center", va="center", fontsize=8.5, color=INK3, rotation=45, transform=ax.transAxes)
        ax.text(0.21, 0.13, "overconfident", ha="center", va="center", fontsize=8.5, color=INK3, rotation=45, transform=ax.transAxes)
        edges = np.linspace(0, 1, 11)
        for lang in ("en", "ru"):
            a = r[lang]; rb = a["reliability"]
            n = np.array(rb["count"]); m = n > 0
            x = np.array(rb["confidence"])[m]; y = np.array(rb["accuracy"])[m]; nn = n[m]
            size = 22 + 260 * np.sqrt(nn / nn.max())
            thin = nn < 10                      # hollow marker: too few items to be evidence
            ax.plot(x, y, "-", lw=1.8, color=C[lang], alpha=0.75, zorder=3)
            ax.scatter(x[~thin], y[~thin], s=size[~thin], color=C[lang], edgecolor="white", linewidth=1.5, zorder=4,
                       label=f"{NAME[lang]}   ECE {a['ece']:.3f}  (noise floor {a['ece_floor']:.3f})")
            ax.scatter(x[thin], y[thin], s=size[thin], facecolor="white", edgecolor=C[lang], linewidth=1.5, zorder=4)
            # count strip: paired thin bars per bin
            off = -0.019 if lang == "en" else 0.019
            centers = (edges[:-1] + edges[1:]) / 2 + off
            axc.bar(centers, n, width=0.034, color=C[lang], zorder=3)
            for cx, cnt in zip(centers, n):
                if cnt > 0:
                    ha = "center" if cnt < 100 else ("right" if lang == "en" else "left")
                    axc.text(cx + (0.006 if ha == "right" else -0.006 if ha == "left" else 0), cnt + n.max() * 0.06,
                             str(int(cnt)), ha=ha, va="bottom", fontsize=6.5, color=INK2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks(edges); ax.tick_params(labelbottom=False)
        ax.set_yticks(edges); ax.set_yticklabels([f"{t:.1f}" if i % 2 == 0 else "" for i, t in enumerate(edges)])
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_ylabel("observed accuracy")
        ax.set_title(f"{TITLE[ds]}\nΔECE (RU − EN) = {_fmt_delta(r['paired']['ece'])}", fontsize=12, loc="left", pad=8)
        ax.legend(loc="lower right", fontsize=9, handletextpad=0.3, handlelength=1.0, borderaxespad=0.3, markerscale=0.5)
        axc.set_xlim(0, 1); axc.set_xticks(edges); axc.set_xticklabels([f"{t:.1f}" for t in edges], fontsize=8.5)
        axc.set_yticks([]); axc.spines["left"].set_visible(False)
        axc.set_ylim(0, axc.get_ylim()[1] * 1.28)
        axc.set_xlabel("stated confidence (p_max, 10 equal-width bins)")
        axc.text(0.0, 1.0, "items per bin", transform=axc.transAxes, fontsize=8, color=INK3, va="bottom")
    fig.suptitle(f"Jev ({res['model']}) — stated confidence vs observed accuracy on the same {res['datasets']['xnli']['n_paired']} items, English vs Russian",
                 fontsize=13, x=0.01, ha="left", color=INK, y=0.995)
    fig.text(0.01, 0.005, "Marker area ∝ items in bin; hollow = fewer than 10 items. Paired bootstrap CI, 1,000 resamples. Headline pass only; English instructions, English option keys. github.com/AHTOOOXA/jev-cyrillic-audit",
             fontsize=8, color=INK3)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def selective_accuracy(res: dict, df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, ds in zip(axes, ("xnli", "massive")):
        r = res["datasets"][ds]
        for lang in ("en", "ru"):
            a = arm(df, ds, lang, 0)
            correct, conf = a["correct"].to_numpy(float), a["confidence"].to_numpy(float)
            cov, sel, _ = M.risk_coverage(correct, conf)
            ax.plot(cov, sel, lw=2, color=C[lang], label=f"{NAME[lang]}   AURC {r[lang]['aurc_confidence']:.3f}", zorder=3)
            for g in GATES:
                gate = r[lang]["gates"][str(g)]
                ax.scatter([gate["coverage"]], [gate["accuracy"]], s=70, color=C[lang], edgecolor="white", linewidth=1.5, zorder=5)
                crowded = gate["coverage"] > 0.93          # near the right edge: label to the left of the dot
                xytext = (-12, 4 if lang == "en" else -22) if crowded else (0, 14 if lang == "en" else -30)
                ax.annotate(f"conf ≥ {g}\n{gate['coverage']:.0%} → {gate['accuracy']:.1%}", (gate["coverage"], gate["accuracy"]),
                            textcoords="offset points", xytext=xytext, ha="right" if crowded else "center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 1.0); ax.set_ylim(0.6, 1.005)
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_xticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])
        ax.set_yticks(np.arange(0.6, 1.001, 0.1)); ax.set_yticklabels([f"{t:.0%}" for t in np.arange(0.6, 1.001, 0.1)])
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("coverage (share of items acted on, most confident first)")
        ax.set_title(TITLE[ds], fontsize=12, loc="left")
        ax.legend(loc="lower left", fontsize=9.5)
    axes[0].set_ylabel("accuracy on the items acted on")
    fig.suptitle(f"Jev ({res['model']}) — accuracy if you only act above a confidence gate, English vs Russian",
                 fontsize=13, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.02, "Ranked by the API's `confidence`; dots mark the vendor's example gates 0.5 and 0.9. Pass 0, n=600 per curve. github.com/AHTOOOXA/jev-cyrillic-audit",
             fontsize=8, color=INK3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(res: dict, out_dir: Path, runs_dir: Path = ROOT / "runs") -> None:
    out_dir.mkdir(exist_ok=True)
    df, _ = load_runs(runs_dir)
    reliability_paired(res, out_dir / "reliability_paired.png")
    selective_accuracy(res, df, out_dir / "selective_accuracy.png")
    print("figures:", sorted(p.name for p in out_dir.glob("*.png")))


if __name__ == "__main__":
    make_figures(json.loads((ROOT / "results.json").read_text()), ROOT / "figures")


# ---------- Study 2 ----------

def panorama_chart(res: dict, out: Path) -> None:
    """dECE(L) and dacc(L) vs the state-token ratio r(L), one point per XNLI language; Belebele-ru as a diamond."""
    langs = res["xnli_languages"]; t = res["tests"]; b = res["belebele"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    panels = (("d_ece", "ΔECE vs English (same items)", t["rq1_ece_vs_tokens"], 1.0),
              ("d_acc", "Δaccuracy vs English (pp)", t["rq2_acc_vs_tokens"], 100.0))
    for ax, (key, ylabel, test, scale) in zip(axes, panels):
        ax.axhline(0, color=INK3, lw=1, ls="--", zorder=1)
        ax.axvline(1, color=GRID, lw=1, zorder=0)
        xs, ys = [], []
        for l, r in sorted(langs.items(), key=lambda kv: kv[1]["tokens"]["state_only_ratio"]):
            x = r["tokens"]["state_only_ratio"]; d = r[key]
            y, lo, hi = d["delta"] * scale, d["ci_low"] * scale, d["ci_high"] * scale
            latin = r["script"] == "latin"
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=C["en"], elinewidth=1.2, alpha=0.6, capsize=2, zorder=2)
            ax.scatter([x], [y], s=64, facecolor=C["en"] if latin else "white", edgecolor=C["en"], linewidth=1.8, zorder=4)
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9, color=INK)
            xs.append(x); ys.append(y)
        # English reference and Belebele-ru
        ax.scatter([1.0], [0.0], s=64, color=INK3, zorder=4); ax.annotate("en", (1.0, 0.0), textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK2)
        bx, bd = b["tokens"]["state_only_ratio"], b[key]
        by = bd["delta"] * scale
        ax.errorbar([bx], [by], yerr=[[by - bd["ci_low"] * scale], [bd["ci_high"] * scale - by]], fmt="none", ecolor=C["ru"], elinewidth=1.2, alpha=0.7, capsize=2, zorder=2)
        ax.scatter([bx], [by], s=80, marker="D", color=C["ru"], edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate("ru · Belebele", (bx, by), textcoords="offset points", xytext=(6, -12), fontsize=9, color=C["ru"])
        ax.set_xscale("log")
        ticks = [1, 1.5, 2, 3, 4, 5]
        ax.set_xlim(0.92, 5.2)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}×" for v in ticks]); ax.minorticks_off()
        ax.set_xlabel("state tokens relative to English (same text)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"Spearman ρ = {test['rho']:+.2f}  [{test['rho_ci_items'][0]:+.2f}, {test['rho_ci_items'][1]:+.2f}]   "
                     f"permutation p = {test['perm_p_one_sided']:.3f}   → {test['verdict']}", fontsize=10.5, loc="left", color=INK)
    fig.suptitle(f"Jev ({res['model']}) — does the tokenization cost of a language predict its calibration loss?  "
                 f"XNLI, 14 languages vs English, same 600 items", fontsize=12.5, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.02, "Filled = Latin script, hollow = other scripts (pre-declared covariate). Bars = paired 95% bootstrap CI. "
             "Belebele (900 long passages, EN vs RU) shown for the task-length comparison. github.com/AHTOOOXA/jev-cyrillic-audit", fontsize=8, color=INK3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure:", out.name)


def neutral_drift_chart(res: dict, out: Path) -> None:
    """Exploratory RQ3: excess `neutral` predictions vs accuracy loss, one point per language."""
    langs = res["xnli_languages"]; t = res["tests"]["rq3_excess_neutral_vs_dacc"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.axhline(0, color=INK3, lw=1, ls="--", zorder=1); ax.axvline(0, color=INK3, lw=1, ls="--", zorder=1)
    for l, r in langs.items():
        x, y = r["excess_neutral"] * 100, r["d_acc"]["delta"] * 100
        latin = r["script"] == "latin"
        ax.errorbar([x], [y], yerr=[[y - r["d_acc"]["ci_low"] * 100], [r["d_acc"]["ci_high"] * 100 - y]], fmt="none", ecolor=C["en"], elinewidth=1.1, alpha=0.5, capsize=2, zorder=2)
        ax.scatter([x], [y], s=64, facecolor=C["en"] if latin else "white", edgecolor=C["en"], linewidth=1.8, zorder=4)
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
    ax.scatter([0], [0], s=64, color=INK3, zorder=4); ax.annotate("en", (0, 0), textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK2)
    xs = np.array([r["excess_neutral"] * 100 for r in langs.values()]); ys = np.array([r["d_acc"]["delta"] * 100 for r in langs.values()])
    b, a = np.polyfit(xs, ys, 1); xx = np.linspace(0, xs.max() * 1.05, 2)
    ax.plot(xx, a + b * xx, color=INK3, lw=1, zorder=1)
    ax.set_xlabel("excess `neutral` predictions vs English (pp of items)")
    ax.set_ylabel("Δaccuracy vs English (pp)")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_title(f"Spearman ρ = {t['rho']:+.2f}, Pearson r = {t['pearson_r']:+.2f}   (exploratory, RQ3)", fontsize=10.5, loc="left", color=INK)
    fig.suptitle(f"Jev ({res['model']}) — the accuracy loss in a language is its drift toward `neutral`\nXNLI, 14 languages vs English, same 600 items", fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.03, f"Slope {b:+.2f} pp accuracy per pp of excess neutral. Filled = Latin script, hollow = other. Bars = paired 95% CI on Δaccuracy. github.com/AHTOOOXA/jev-cyrillic-audit", fontsize=8, color=INK3)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("figure:", out.name)


# ---------- Study 3 ----------

def mechanism_charts(res: dict, out_dir: Path) -> None:
    r1, r2, r3 = res["h1_neutral_drift"], res["h2_binary_framing"], res["h3_sib200"]
    langs = r1["languages"]; t = r1["test"]
    # --- 1. fresh-item drift line, Study-2 points as a faint reference
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.axhline(0, color=INK3, lw=1, ls="--", zorder=1); ax.axvline(0, color=INK3, lw=1, ls="--", zorder=1)
    ref = ROOT / "results2.json"
    if ref.exists():
        s2 = json.loads(ref.read_text())["xnli_languages"]
        ax.scatter([v["excess_neutral"] * 100 for v in s2.values()], [v["d_acc"]["delta"] * 100 for v in s2.values()],
                   s=40, facecolor="none", edgecolor=INK3, linewidth=1, alpha=0.6, zorder=2, label="Study 2 (test items, exploratory)")
    for l, r in langs.items():
        x, y = r["excess_neutral"] * 100, r["d_acc"]["delta"] * 100
        ax.errorbar([x], [y], yerr=[[y - r["d_acc"]["ci_low"] * 100], [r["d_acc"]["ci_high"] * 100 - y]], fmt="none", ecolor=C["en"], elinewidth=1.1, alpha=0.5, capsize=2, zorder=3)
        ax.scatter([x], [y], s=64, facecolor=C["en"] if r["script"] == "latin" else "white", edgecolor=C["en"], linewidth=1.8, zorder=4)
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color=INK)
    ax.scatter([], [], s=64, color=C["en"], label="Study 3 (fresh validation items, pre-registered)")
    ax.scatter([0], [0], s=64, color=INK3, zorder=4); ax.annotate("en", (0, 0), textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK2)
    xs = np.array([r["excess_neutral"] * 100 for r in langs.values()]); ys = np.array([r["d_acc"]["delta"] * 100 for r in langs.values()])
    b, a = np.polyfit(xs, ys, 1); xx = np.linspace(0, xs.max() * 1.05, 2); ax.plot(xx, a + b * xx, color=INK3, lw=1, zorder=1)
    ax.set_xlabel("excess `neutral` predictions vs English (pp of items)"); ax.set_ylabel("Δaccuracy vs English (pp)")
    ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(f"ρ = {t['rho']:+.2f} [{t['rho_ci_items'][0]:+.2f}, {t['rho_ci_items'][1]:+.2f}], permutation p = {t['perm_p_one_sided']:.4f}; "
                 f"{t['pooled_lost_to_neutral']:.0%} of lost items → neutral  → {t['verdict']}", fontsize=9.5, loc="left", color=INK)
    fig.suptitle(f"Jev ({res['model']}) — neutral drift replicates on 600 fresh XNLI items\n14 languages vs English", fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.03, f"Slope {b:+.2f} pp per pp. Filled = Latin script, hollow = other. Bars = paired 95% CI. github.com/AHTOOOXA/jev-cyrillic-audit", fontsize=8, color=INK3)
    fig.tight_layout(); fig.savefig(out_dir / "mechanism_fresh_drift.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # --- 2. recovery: recall on gold-entailment items, 3-way vs binary, EN vs L
    L = list(r2["languages"]); n = len(L); w = 0.2; xi = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    series = [("3-way · English", [r2["languages"][l]["recall_ent_3way"]["en"] for l in L], INK3, 1.0),
              ("3-way · language", [r2["languages"][l]["recall_ent_3way"]["lang"] for l in L], C["en"], 1.0),
              ("binary · English", [r2["languages"][l]["recall_ent_binary"]["en"] for l in L], INK3, 0.45),
              ("binary · language", [r2["languages"][l]["recall_ent_binary"]["lang"] for l in L], C["ru"], 1.0)]
    for j, (lab, vals, col, alpha) in enumerate(series):
        ax.bar(xi + (j - 1.5) * w, vals, width=w - 0.02, color=col, alpha=alpha, label=lab, zorder=3)
    for i, l in enumerate(L):
        r = r2["languages"][l]
        txt = "n/a" if r["recovery"] is None else f"recovery {r['recovery']:+.0%}"
        ax.text(i, 1.02, txt, ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xi); ax.set_xticklabels(L); ax.set_ylim(0, 1.1); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("recall on gold-entailment items (n=200)"); ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, ncol=4)
    ci = r2["mean_recovery_ci_items"]
    ax.set_title(f"Mean recovery of the entailment gap when `neutral` is removed: {r2['mean_recovery']:+.0%} [{ci[0]:+.0%}, {ci[1]:+.0%}]  → {r2['verdict']}", fontsize=10, loc="left", color=INK)
    fig.suptitle(f"Jev ({res['model']}) — does the loss survive without an \"undetermined\" option?", fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.09, "Same 600 fresh items asked two ways: 3-way (entailment / neutral / contradiction) and binary (entailment / not). github.com/AHTOOOXA/jev-cyrillic-audit", fontsize=8, color=INK3)
    fig.tight_layout(); fig.savefig(out_dir / "mechanism_recovery.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # --- 3. tasks: dacc per language, XNLI-fresh vs SIB-200
    order = sorted(langs, key=lambda l: langs[l]["d_acc"]["delta"])
    fig, ax = plt.subplots(figsize=(9, 5.2)); yi = np.arange(len(order))
    for i, l in enumerate(order):
        for key, src, col, mk in (("XNLI (inference)", r1["languages"][l], C["en"], "o"), ("SIB-200 (topic selection)", r3["languages"][l], C["ru"], "D")):
            d = src["d_acc"]; y = i + (0.12 if mk == "D" else -0.12)
            ax.plot([d["ci_low"] * 100, d["ci_high"] * 100], [y, y], color=col, lw=1.2, alpha=0.5, zorder=2)
            ax.scatter([d["delta"] * 100], [y], s=48, color=col, marker=mk, edgecolor="white", linewidth=0.8, zorder=4, label=key if i == 0 else None)
    ax.axvline(0, color=INK3, lw=1, ls="--", zorder=1)
    ax.set_yticks(yi); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel("Δaccuracy vs English (pp), paired 95% CI"); ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=2)
    ax.set_title(f"median Δacc: XNLI {r3['median_d_acc_xnli_fresh']*100:+.1f} pp vs SIB-200 {r3['median_d_acc_sib200']*100:+.1f} pp  → {r3['verdict']}", fontsize=10, loc="left", color=INK)
    fig.suptitle(f"Jev ({res['model']}) — the same 14 languages on an inference task and a selection task", fontsize=12, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.08, "XNLI: 600 fresh items (3-way). SIB-200: 204 parallel sentences, 7 topics. github.com/AHTOOOXA/jev-cyrillic-audit", fontsize=8, color=INK3)
    fig.tight_layout(); fig.savefig(out_dir / "mechanism_tasks.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("figures:", "mechanism_fresh_drift.png mechanism_recovery.png mechanism_tasks.png")

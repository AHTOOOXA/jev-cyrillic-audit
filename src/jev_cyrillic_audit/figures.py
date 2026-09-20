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

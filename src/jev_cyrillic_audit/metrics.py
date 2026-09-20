"""Calibration and accuracy metrics — numpy only, ~150 lines, unit-tested on synthetic data.

Definitions follow research §4: 10 equal-width left-closed bins over [0, 1] with exact decimal
edges and a closed top bin, sample-weighted |acc − conf| per bin; paired bootstrap on RU−EN.
"""

from __future__ import annotations

import numpy as np

N_BINS = 10
N_BOOT = 1000
N_SIM = 1000
SEED = 0


# ---------- accuracy ----------

def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred) -> float:
    """Per-class F1 averaged with equal weight over the classes that occur in gold."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


# ---------- calibration ----------

def bin_index(conf, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-width bins over [0, 1], left-closed: bin b covers [b/n, (b+1)/n); the top bin is closed at 1.

    Edges are exact decimals (computed as floor(conf·n) after rounding away float noise), so a
    2-decimal API value such as 0.90 lands in the 0.9–1.0 bin. netcal / np.histogramdd use the same
    left-closed rule but build edges with np.linspace, whose float noise (0.30000000000000004,
    0.7000000000000001) pushes some exact ties into the lower bin; the tests show the two agree to
    1e-9 once that artefact is removed. jev-benchmarks uses the same left-closed convention.
    """
    conf = np.asarray(conf, dtype=float)
    return np.clip(np.floor(np.round(conf * n_bins, 9)).astype(int), 0, n_bins - 1)


def reliability_bins(correct, conf, n_bins: int = N_BINS) -> dict:
    """Per-bin mean confidence, observed accuracy and count (empty bins have count 0)."""
    correct, conf = np.asarray(correct, dtype=float), np.asarray(conf, dtype=float)
    idx = bin_index(conf, n_bins)
    n = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.bincount(idx, weights=correct, minlength=n_bins) / n
        mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / n
    return {"count": n, "accuracy": acc, "confidence": mean_conf}


def ece(correct, conf, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: sum_b (n_b / n) · |acc_b − conf_b|."""
    b = reliability_bins(correct, conf, n_bins)
    m = b["count"] > 0
    return float(np.sum(b["count"][m] / b["count"].sum() * np.abs(b["accuracy"][m] - b["confidence"][m])))


def ece_floor(conf, n_bins: int = N_BINS, n_sim: int = N_SIM, seed: int = SEED) -> float:
    """Mean ECE a PERFECTLY calibrated model scores on this confidence vector (finite-sample bias)."""
    rng = np.random.default_rng(seed)
    conf = np.asarray(conf, dtype=float)
    return float(np.mean([ece(rng.random(conf.size) < conf, conf, n_bins) for _ in range(n_sim)]))


# ---------- selective prediction ----------

def risk_coverage(correct, score):
    """Sort by `score` descending; selective accuracy and risk at every coverage level."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    c = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, c.size + 1)
    coverage = k / c.size
    sel_acc = np.cumsum(c) / k
    return coverage, sel_acc, 1.0 - sel_acc


def aurc(correct, score) -> float:
    coverage, _, risk = risk_coverage(correct, score)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(risk, coverage))


def aurc_anchors(acc: float) -> dict:
    """AURC of a perfect ranker and of an uninformative one, for the same accuracy."""
    return {"optimal": float((1 - acc) + acc * np.log(acc)) if acc > 0 else 1.0, "random": float(1 - acc)}


def coverage_at_gate(correct, score, threshold: float) -> dict:
    """Share of items at or above the gate and the accuracy among them."""
    correct, score = np.asarray(correct, dtype=float), np.asarray(score, dtype=float)
    m = score >= threshold
    return {"coverage": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else float("nan"), "n": int(m.sum())}


# ---------- bootstrap ----------

def boot_ci(stat_fn, *arrays, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over items; the same resampled indices are applied to every array."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a) for a in arrays]
    n = arrs[0].shape[0]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        vals[i] = stat_fn(*[a[k] for a in arrs])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_delta(stat_fn, arrays_a, arrays_b, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> dict:
    """stat(A) − stat(B) with a paired bootstrap CI: resample item indices once, apply to both arms."""
    rng = np.random.default_rng(seed)
    A = [np.asarray(a) for a in arrays_a]
    B = [np.asarray(b) for b in arrays_b]
    n = A[0].shape[0]
    assert all(x.shape[0] == n for x in A + B), "paired arms must have the same items in the same order"
    obs = stat_fn(*A) - stat_fn(*B)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        vals[i] = stat_fn(*[a[k] for a in A]) - stat_fn(*[b[k] for b in B])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(obs), "ci_low": float(lo), "ci_high": float(hi)}


# ---------- determinism ----------

def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == l) * np.mean(b == l) for l in labels))
    return 1.0 if pe == 1.0 else float((po - pe) / (1 - pe))


def determinism(pred0, pred1, pmax0, pmax1) -> dict:
    pred0, pred1 = np.asarray(pred0), np.asarray(pred1)
    d = np.abs(np.asarray(pmax0, dtype=float) - np.asarray(pmax1, dtype=float))
    return {
        "flip_rate": float(np.mean(pred0 != pred1)),
        "kappa": cohen_kappa(pred0, pred1),
        "mean_abs_dpmax": float(d.mean()),
        "max_abs_dpmax": float(d.max()),
        "share_identical_pmax": float(np.mean(d == 0)),
    }

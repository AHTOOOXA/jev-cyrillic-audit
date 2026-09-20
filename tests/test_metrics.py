import numpy as np
import pytest

from jev_cyrillic_audit import metrics as M


def _perfectly_calibrated(n, seed=1):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.34, 1.0, n)
    correct = rng.random(n) < conf
    return correct, conf


def test_accuracy_and_macro_f1_on_balanced_random_binary():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000); p = rng.integers(0, 2, 20000)
    assert abs(M.accuracy(y, p) - 0.5) < 0.02 and abs(M.macro_f1(y, p) - 0.5) < 0.02


def test_macro_f1_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 500); p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 5, 500))
    assert abs(M.macro_f1(y, p) - sk.f1_score(y, p, average="macro")) < 1e-12


def test_ece_perfectly_calibrated_is_near_its_floor():
    correct, conf = _perfectly_calibrated(2000)
    e, floor = M.ece(correct, conf), M.ece_floor(conf, n_sim=300)
    assert e < 2.5 * floor and floor < 0.03


def test_ece_maximally_overconfident_equals_one_minus_accuracy():
    rng = np.random.default_rng(2)
    correct = rng.random(1000) < 0.8
    conf = np.ones(1000)
    assert abs(M.ece(correct, conf) - (1 - correct.mean())) < 1e-12


def test_bin_edges_left_closed_exact_decimals_top_bin_closed():
    idx = M.bin_index([0.0, 0.09, 0.1, 0.11, 0.3, 0.6, 0.7, 0.89, 0.9, 0.91, 0.99, 1.0])
    assert idx.tolist() == [0, 0, 1, 1, 3, 6, 7, 8, 9, 9, 9, 9]


def test_ece_matches_netcal_to_1e9():
    netcal = pytest.importorskip("netcal.metrics")
    ref = netcal.ECE(bins=10, equal_intervals=True)
    rng = np.random.default_rng(5)
    edges = np.round(np.arange(0, 1.01, 0.1), 2)
    for n in (60, 600, 5000):
        # continuous confidences: no ties on edges, the formula itself must agree
        conf = rng.uniform(0.3, 1.0, n); correct = rng.random(n) < conf * 0.9
        assert abs(M.ece(correct, conf) - ref.measure(conf, correct.astype(int))) < 1e-9
        # 2-decimal confidences (the API's precision): agree once netcal's float-noise edge
        # artefact is removed by nudging exact ties up by 1e-9 (netcal is left-closed like us)
        conf2 = np.round(rng.uniform(0.3, 1.0, n), 2); correct2 = rng.random(n) < conf2 * 0.9
        nudged = np.where(np.isin(conf2, edges) & (conf2 < 1), conf2 + 1e-9, conf2)
        assert abs(M.ece(correct2, conf2) - ref.measure(nudged, correct2.astype(int))) < 1e-9


def test_floor_grows_as_n_shrinks():
    rng = np.random.default_rng(7)
    conf_big, conf_small = rng.uniform(0.5, 1, 2000), rng.uniform(0.5, 1, 60)
    assert M.ece_floor(conf_small, n_sim=200) > M.ece_floor(conf_big, n_sim=200)


def test_aurc_between_anchors_and_perfect_ranking_hits_optimal():
    rng = np.random.default_rng(8)
    correct = rng.random(5000) < 0.9
    perfect = correct.astype(float) + rng.random(5000) * 1e-3   # correct items ranked first
    a = M.aurc(correct, perfect); anchors = M.aurc_anchors(correct.mean())
    assert abs(a - anchors["optimal"]) < 0.01
    random_score = rng.random(5000)
    assert abs(M.aurc(correct, random_score) - anchors["random"]) < 0.02


def test_paired_delta_ci_covers_truth_and_is_tighter_than_marginals():
    rng = np.random.default_rng(9)
    n = 600
    base = rng.random(n) < 0.85
    flip = rng.random(n) < 0.05
    ru = np.where(flip, ~base, base)      # RU flips 5% of items; E[delta] = 0.05·(0.15 − 0.85) = −3.5 pp
    d = M.paired_delta(M.accuracy, [ru, np.ones(n)], [base, np.ones(n)])
    assert d["ci_low"] <= -0.035 <= d["ci_high"]
    assert abs(d["delta"] - (ru.mean() - base.mean())) < 1e-12
    lo, hi = M.boot_ci(lambda a, b: M.accuracy(a, b), base, np.ones(n))
    assert (d["ci_high"] - d["ci_low"]) < (hi - lo)   # pairing cancels item difficulty


def test_determinism_table():
    p0 = np.array(["a", "b", "a", "c"]); p1 = np.array(["a", "b", "c", "c"])
    d = M.determinism(p0, p1, [1, .9, .8, .7], [1, .8, .8, .6])
    assert d["flip_rate"] == 0.25 and d["share_identical_pmax"] == 0.5 and abs(d["max_abs_dpmax"] - 0.1) < 1e-12
    assert M.cohen_kappa(p0, p0) == 1.0


def test_coverage_at_gate():
    r = M.coverage_at_gate([1, 1, 0, 1], [0.95, 0.5, 0.91, 0.2], 0.9)
    assert r == {"coverage": 0.5, "accuracy": 0.5, "n": 2}

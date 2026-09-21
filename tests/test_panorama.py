import numpy as np
import pytest

from jev_cyrillic_audit import panorama as P


def test_rankdata_averages_ties_and_spearman_matches_scipy_when_available():
    assert P.rankdata([3, 1, 2, 2]).tolist() == [4.0, 1.0, 2.5, 2.5]
    rng = np.random.default_rng(0)
    x = rng.normal(size=14); y = x + rng.normal(size=14)
    sp = pytest.importorskip("scipy.stats")
    assert abs(P.spearman(x, y) - sp.spearmanr(x, y).statistic) < 1e-12


def test_permutation_p_is_small_for_a_perfect_monotone_relation_and_large_for_noise():
    x = np.arange(14, dtype=float)
    assert P.perm_p_one_sided(x, x, +1, n_perm=2000) < 0.001
    assert P.perm_p_one_sided(x, -x, +1, n_perm=2000) > 0.99
    rng = np.random.default_rng(1)
    ps = [P.perm_p_one_sided(x, rng.permutation(x), +1, n_perm=500, seed=i) for i in range(20)]
    assert 0.2 < np.mean(ps) < 0.8


def test_decision_rule_branches():
    assert P.decide(0.7, 0.01, [0.2, 0.9], +1) == "SUPPORTED"
    assert P.decide(0.1, 0.4, [-0.4, 0.5], +1) == "NOT SUPPORTED"
    assert P.decide(0.5, 0.03, [-0.1, 0.8], +1) == "AMBIGUOUS at 14 languages"   # p small but CI spans 0
    assert P.decide(-0.7, 0.01, [-0.9, -0.2], -1) == "SUPPORTED"

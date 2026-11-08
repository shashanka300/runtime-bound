import numpy as np
import pytest

from rbcore import stats


def test_bootstrap_ci_contains_mean_and_is_deterministic() -> None:
    x = np.random.default_rng(1).binomial(1, 0.4, size=200)
    e1 = stats.bootstrap_ci(x, seed=7, n_resamples=2000)
    e2 = stats.bootstrap_ci(x, seed=7, n_resamples=2000)
    assert e1 == e2
    assert e1.lo < e1.point < e1.hi
    # normal-approx half-width sqrt(p(1-p)/n)*1.96 ~ 0.068
    assert 0.05 < (e1.hi - e1.lo) / 2 < 0.09


def test_stratified_bootstrap_keeps_constant_strata_exact() -> None:
    # stratum A all 1s, stratum B all 0s: every stratified resample has the same mean
    x = np.array([1] * 30 + [0] * 10)
    strata = np.array(["a"] * 30 + ["b"] * 10)
    e = stats.bootstrap_ci(x, strata=strata, n_resamples=500)
    assert e.lo == e.hi == pytest.approx(0.75)


def test_paired_diff_sign_and_shape_check() -> None:
    a = np.zeros(50)
    b = np.r_[np.ones(20), np.zeros(30)]
    e = stats.paired_diff(a, b, n_resamples=1000)
    assert e.point == pytest.approx(0.4) and e.lo > 0
    with pytest.raises(ValueError):
        stats.paired_diff(a, b[:-1])


def test_mcnemar_exact_matches_binomial() -> None:
    a = np.array([1] * 10 + [0] * 2 + [1] * 5 + [0] * 5, bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 5 + [0] * 5, bool)
    r = stats.mcnemar_exact(a, b)
    assert (r.only_a, r.only_b) == (10, 2)
    assert r.p_value == pytest.approx(0.03857421875)  # 2 * P(X<=2 | n=12, p=.5)
    assert stats.mcnemar_exact(a, a).p_value == 1.0


def test_wilson_ci_reference_value() -> None:
    e = stats.wilson_ci(8, 10)
    assert (e.lo, e.hi) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9433, abs=1e-4))

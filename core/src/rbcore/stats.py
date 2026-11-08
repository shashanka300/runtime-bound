"""Statistics for paired, per-task agent evaluations.

Conventions (matching the research plan): tasks are the unit of resampling; systems
are compared on the *same* tasks (paired); intervals are percentile bootstrap 95%
CIs with a fixed seed; paired binary outcomes use the exact McNemar test. Seeds of
the same task are averaged before resampling tasks, so seed noise does not inflate n.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sps

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def bootstrap_ci(
    values: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
    strata: ArrayLike | None = None,
) -> Estimate:
    """Percentile bootstrap. ``strata`` (e.g. repo per task) resamples within each stratum,
    keeping the repo mix fixed, as in the stratified bootstrap of Agarwal et al. (2021)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    else:
        s = np.asarray(strata)
        groups = [np.flatnonzero(s == g) for g in np.unique(s)]
        idx = np.concatenate(
            [g[rng.integers(0, len(g), size=(n_resamples, len(g)))] for g in groups], axis=1
        )
    boots = np.apply_along_axis(statistic, 1, x[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(statistic(x)), float(lo), float(hi), len(x))


def paired_diff(
    a: ArrayLike,
    b: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    seed: int = 0,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Estimate:
    """CI of mean(b - a) over tasks, where a[i] and b[i] are the same task."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    if a_.shape != b_.shape:
        raise ValueError("paired arrays must align task-for-task")
    return bootstrap_ci(b_ - a_, strata=strata, seed=seed, n_resamples=n_resamples)


@dataclass(frozen=True)
class McNemar:
    only_a: int
    only_b: int
    p_value: float


def mcnemar_exact(a: ArrayLike, b: ArrayLike) -> McNemar:
    """Exact (binomial) McNemar test on paired resolved/unresolved outcomes."""
    a_, b_ = np.asarray(a, bool), np.asarray(b, bool)
    only_a = int(np.sum(a_ & ~b_))
    only_b = int(np.sum(~a_ & b_))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(sps.binomtest(only_a, n, 0.5).pvalue)
    return McNemar(only_a, only_b, p)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Estimate:
    """Wilson score interval for a single resolved rate (good at small n / extreme p)."""
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    z = sps.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Estimate(p, float(centre - half), float(centre + half), n)

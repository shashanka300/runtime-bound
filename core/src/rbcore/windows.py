"""Attribute telemetry to time windows (phases, tasks, benchmark cells).

All sources share the wall-clock ``t_ns`` column, so attribution is an interval join:
a sample belongs to every window with ``t_start_ns <= t_ns <= t_end_ns``.

Cumulative counters (NVML energy, vLLM ``*_total`` / histogram ``_sum``/``_count``)
are attributed as last-minus-first inside the window. At 1 Hz sampling this
under-counts by at most one interval per window edge, so windows should be long
relative to the interval (benchmark cells and agent tasks are tens of seconds+).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

WINDOW_KEYS = ("t_start_ns", "t_end_ns")


def in_windows(samples: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Samples tagged with the columns of each window they fall in."""
    return windows.join_where(
        samples,
        pl.col("t_ns") >= pl.col("t_start_ns"),
        pl.col("t_ns") <= pl.col("t_end_ns"),
    )


def per_window(
    samples: pl.DataFrame,
    windows: pl.DataFrame,
    aggs: Sequence[pl.Expr],
    by: Sequence[str] = (),
) -> pl.DataFrame:
    """Aggregate samples per window. Windows with no samples are kept (nulls)."""
    keys = [c for c in windows.columns]
    joined = in_windows(samples, windows)
    out = joined.group_by([*keys, *by], maintain_order=True).agg(aggs)
    return windows.join(out, on=keys, how="left", nulls_equal=True)


def gpu_energy_j(gpu: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Joules per window from the NVML cumulative energy counter, summed over GPUs."""
    per_gpu = per_window(
        gpu,
        windows,
        [(pl.col("energy_mj").max() - pl.col("energy_mj").min()).alias("mj")],
        by=["gpu"],
    )
    keys = list(windows.columns)
    return per_gpu.group_by(keys, maintain_order=True).agg(
        (pl.col("mj").sum() / 1000.0).alias("gpu_energy_j")
    )


def counter_delta(vllm: pl.DataFrame, windows: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Increase of a vLLM cumulative metric inside each window (summed over label sets)."""
    rows = vllm.filter(pl.col("metric") == metric)
    per_label = per_window(
        rows, windows, [(pl.col("value").max() - pl.col("value").min()).alias("d")], by=["labels"]
    )
    keys = list(windows.columns)
    return per_label.group_by(keys, maintain_order=True).agg(pl.col("d").sum().alias(metric))

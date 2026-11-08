"""Resolved rate vs wall-clock across (model, thinking budget, attempts) — H1.

An *attempt* is an independent fresh-context session of the same configuration (run seed
0..n-1, executed in that order). ``k`` attempts are the first ``k`` seeds, scored two ways:

- ``deploy``: stop at the first attempt that submitted a non-empty patch and grade that
  patch; if none submitted, grade the last non-empty patch. Implementable at eval time
  (no test oracle). Wall-clock = agent seconds of the attempts actually run.
- ``oracle`` (pass@k): resolved if any of the k attempts resolved. Upper bound; wall-clock
  = all k attempts.

Pre-registered comparisons (31B only; see README "Design change"): configurations are
compared on the same tasks with a paired bootstrap of the resolved difference (stratified
by repo) and an exact McNemar test; a difference counts only if its 95% CI excludes 0.
``h1_test`` (two models at matched wall-clock) is kept for multi-model sweeps.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from rbcore import stats

COND = ["model", "thinking"]


def attempts_table(results: pl.DataFrame) -> pl.DataFrame:
    """results must carry model, thinking, instance_id, repo, seed, resolved, submitted,
    patch_chars, agent_seconds. Returns one row per (model, thinking, task) with the
    attempts as seed-ordered lists."""
    return (
        results.sort("seed")
        .group_by([*COND, "instance_id", "repo"], maintain_order=True)
        .agg(
            pl.col("resolved"),
            pl.col("submitted"),
            pl.col("patch_chars"),
            pl.col("agent_seconds").fill_null(0.0),
            pl.col("seed"),
        )
    )


def score_k(row: dict[str, Any], k: int) -> dict[str, Any]:
    res, sub = row["resolved"][:k], row["submitted"][:k]
    chars, secs = row["patch_chars"][:k], row["agent_seconds"][:k]
    chosen = next((i for i in range(len(res)) if sub[i] and (chars[i] or 0) > 0), None)
    if chosen is None:
        nonempty = [i for i in range(len(res)) if (chars[i] or 0) > 0]
        chosen = nonempty[-1] if nonempty else len(res) - 1
        ran = len(res)
    else:
        ran = chosen + 1
    return {
        "deploy": bool(res[chosen]) if res else False,
        "deploy_seconds": float(sum(secs[:ran])),
        "oracle": any(res),
        "oracle_seconds": float(sum(secs)),
        "attempts_run": ran,
    }


def per_task(attempts: pl.DataFrame, ks: list[int]) -> pl.DataFrame:
    rows = []
    for r in attempts.iter_rows(named=True):
        n = len(r["resolved"])
        for k in ks:
            if k <= n:
                rows.append(
                    {**{c: r[c] for c in [*COND, "instance_id", "repo"]}, "k": k, **score_k(r, k)}
                )
    return pl.DataFrame(rows)


def configs(tasks: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Per (model, thinking, k): deploy/oracle rates with CIs and mean wall-clock."""
    out = []
    for (model, thinking, k), g in tasks.group_by([*COND, "k"], maintain_order=True):
        d = stats.bootstrap_ci(g["deploy"].cast(int), strata=g["repo"], seed=seed)
        o = stats.bootstrap_ci(g["oracle"].cast(int), strata=g["repo"], seed=seed)
        out.append(
            {
                "model": model,
                "thinking": thinking,
                "k": k,
                "n": g.height,
                "deploy": d.point,
                "deploy_lo": d.lo,
                "deploy_hi": d.hi,
                "oracle": o.point,
                "oracle_lo": o.lo,
                "oracle_hi": o.hi,
                "deploy_seconds": g["deploy_seconds"].mean(),
                "oracle_seconds": g["oracle_seconds"].mean(),
            }
        )
    return pl.DataFrame(out).sort("deploy_seconds")


def pareto(cfg: pl.DataFrame, x: str = "deploy_seconds", y: str = "deploy") -> pl.DataFrame:
    """Configurations not dominated (no other is at least as fast and strictly better)."""
    best, keep = -1.0, []
    for r in cfg.sort(x).iter_rows(named=True):
        if r[y] > best:
            keep.append(r)
            best = r[y]
    return pl.DataFrame(keep)


def h1_test(
    tasks: pl.DataFrame, cfg: pl.DataFrame, large: str, small: str, seed: int = 0
) -> tuple[str, pl.DataFrame]:
    budgets = sorted(set(cfg["deploy_seconds"].to_list()))
    rows = []
    for b in budgets:
        fit = cfg.filter(pl.col("deploy_seconds") <= b + 1e-9)
        pick = {}
        for m in (large, small):
            cand = fit.filter(pl.col("model") == m).sort("deploy", descending=True)
            if cand.height:
                pick[m] = cand.row(0, named=True)
        if len(pick) < 2:
            continue

        def sel(m: str, pick: dict[str, dict[str, Any]] = pick) -> pl.DataFrame:
            p = pick[m]
            return tasks.filter(
                (pl.col("model") == m)
                & (pl.col("thinking") == p["thinking"])
                & (pl.col("k") == p["k"])
            ).select("instance_id", "repo", pl.col("deploy").alias(m))

        pair = sel(large).join(sel(small).drop("repo"), on="instance_id")
        diff = stats.paired_diff(
            pair[small].cast(int), pair[large].cast(int), strata=pair["repo"], seed=seed
        )  # large - small
        rows.append(
            {
                "budget_s": b,
                "large_cfg": f"t{pick[large]['thinking']}/k{pick[large]['k']}",
                "small_cfg": f"t{pick[small]['thinking']}/k{pick[small]['k']}",
                "large_rate": pick[large]["deploy"],
                "small_rate": pick[small]["deploy"],
                "diff": diff.point,
                "diff_lo": diff.lo,
                "diff_hi": diff.hi,
                "n": diff.n,
            }
        )
    table = pl.DataFrame(rows)
    if table.is_empty():
        return "INCONCLUSIVE", table
    if (table["diff_lo"] > 0).all():
        return "FALSIFIED", table
    if ((table["small_rate"] >= table["large_rate"]) & (table["diff_lo"] <= 0)).any():
        return "SUPPORTED", table
    return "INCONCLUSIVE", table


def paired(
    tasks: pl.DataFrame, a: dict[str, Any], b: dict[str, Any], metric: str = "deploy", seed: int = 0
) -> dict[str, Any]:
    """b - a on the same tasks; a/b select rows by model, thinking and k."""

    def sel(c: dict[str, Any], name: str) -> pl.DataFrame:
        return tasks.filter(
            (pl.col("model") == c["model"])
            & (pl.col("thinking") == c["thinking"])
            & (pl.col("k") == c["k"])
        ).select(
            "instance_id",
            "repo",
            pl.col(metric).alias(name),
            pl.col(f"{metric}_seconds").alias(f"{name}_s"),
        )

    pair = sel(a, "a").join(sel(b, "b").drop("repo"), on="instance_id")
    d = stats.paired_diff(pair["a"].cast(int), pair["b"].cast(int), strata=pair["repo"], seed=seed)
    mc = stats.mcnemar_exact(pair["a"], pair["b"])
    return {
        "a": f"t{a['thinking']}/k{a['k']}",
        "b": f"t{b['thinking']}/k{b['k']}",
        "rate_a": float(pair["a"].mean()),
        "rate_b": float(pair["b"].mean()),
        "seconds_a": float(pair["a_s"].mean()),
        "seconds_b": float(pair["b_s"].mean()),
        "diff": d.point,
        "lo": d.lo,
        "hi": d.hi,
        "n": d.n,
        "only_a": mc.only_a,
        "only_b": mc.only_b,
        "p_mcnemar": mc.p_value,
        "significant": d.lo > 0 or d.hi < 0,
    }

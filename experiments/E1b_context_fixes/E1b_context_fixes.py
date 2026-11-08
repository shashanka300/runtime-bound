# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: challeenge (py3.12)
#     language: python
#     name: challeenge
# ---

# %% [markdown]
# # E1b — Context-budget fixes for the starter agent (31B)
#
# **Why.** E1/E2 found 93% of starter failures are context exhaustion: vLLM rejects a request
# when prompt + `max_output_tokens` (16384) > 32768, so the usable prompt is 16k, and runs die
# while still localizing. This ablation measures what cheap fixes recover.
#
# | Arm | Change (vs parent) | Submittable |
# |---|---|---|
# | A | starter as-is | yes |
# | B | A + `max_output_tokens` = thinking + 4096 (usable prompt 16k → 24k) | yes |
# | C | B + prompt guidance: grep then small reads; scratch via `run_command` in `/tmp` | yes |
# | D | B + fresh-context retry on the same workspace after a context death | diagnostic |
# | E | B + Gemma's recommended sampling (T 1.0, top_p 0.95, top_k 64); added 2026-09-26 | yes |
#
# **Pre-registered comparisons** (paired on the same task × seed, 95% bootstrap CI stratified
# by repo, exact McNemar): B−A, C−B, D−B, C−A. A fix counts only if its CI excludes 0.

# %% [markdown]
# ## 0 · Parameters

# %% tags=["parameters"]
CONFIG_OVERRIDES: dict = {}

# %% [markdown]
# ## 1 · Setup & provenance

# %%
from pathlib import Path

import polars as pl
from IPython.display import Markdown, display
from src.arms import context_retries, make_arm
from src.params import Params

from rbcore import RunContext, load_config, plotting, stats
from rbcore.paths import data_root
from swebox import sandbox as sb
from swebox.analysis import load_results
from swebox.evaluate import EvalSpec, evaluate
from swebox.submission import ServingTarget
from swebox.tasks import load_tasks, select
from swebox.tools import Budget

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
tasks = select(load_tasks(), P.subset)
task_ids = {t.instance_id for t in tasks}
limits = sb.Limits(cfg.sandbox.mem_gib or 8, cfg.sandbox.cpus or 4)
reused = {**({"A": P.reuse_arm_a} if P.reuse_arm_a else {}), **P.reuse}
run_arms = [a for a in P.arms if a not in reused] + ([] if "A" in reused else ["A"])
display(
    Markdown(
        f"config `{cfg.hash()}` · {len(tasks)} tasks × seeds {P.seeds} · running arms "
        f"{run_arms} · reused {reused or 'none'} · "
        f"**{len(run_arms) * len(tasks) * len(P.seeds)} sessions**"
    )
)

# %% [markdown]
# ## 2 · Preflight
# The starter's LoRA adapters must be served (all arms keep them, so only the named factor
# changes): `bash infra/vllm/serve_submission.sh`.

# %%
import httpx

served = {
    m["id"]
    for m in httpx.get(f"{cfg.serving.base_url}/v1/models", timeout=5)
    .raise_for_status()
    .json()["data"]
}
if any(a not in ("F", "G", "H") for a in run_arms):  # F/G/H drop the adapters
    assert {"main_lora", "tool_lora"} <= served, f"adapters not served: {served}"
reuse: dict[str, pl.DataFrame] = {}
for arm, src_dir in reused.items():
    reuse[arm] = load_results(Path(src_dir)).filter(
        pl.col("instance_id").is_in(task_ids) & pl.col("seed").is_in(P.seeds)
    )
    missing = len(task_ids) * len(P.seeds) - reuse[arm].height
    display(Markdown(f"reused arm {arm} rows: {reuse[arm].height} (missing {missing})"))
    assert missing == 0, f"reused arm {arm} lacks some (task, seed) pairs; run it instead"
dc = sb.client()
dc.ping()

# %% [markdown]
# ## 3 · Open the run

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 8 with run.finish()
out_root = Path(P.results_dir) if P.results_dir else run.artifacts / "swebox"
run.event("results_dir", path=str(out_root), resumed=P.results_dir is not None)
print(run.dir)

# %% [markdown]
# ## 4 · Run the arms
# Resumable: finished (task, seed) pairs are skipped per arm.

# %%
base = data_root() / P.base_submission
for arm in run_arms:
    sub = make_arm(base, run.artifacts / "submissions" / arm, arm, P.output_reserve)
    spec = EvalSpec(
        submission=sub,
        limits=limits,
        budget=Budget(
            time_minutes=P.time_minutes, context_retries=context_retries(arm, P.d_context_retries)
        ),
        target=ServingTarget(
            api_base=f"{cfg.serving.base_url}/v1",
            served_model=cfg.serving.model,
            thinking_translation=P.thinking_translation,
        ),
        seeds=tuple(P.seeds),
        concurrency=P.concurrency,
        tag=f"arm_{arm}",
    )
    with run.phase("arm", arm=arm):
        rows = await evaluate(tasks, spec, out_root / f"arm_{arm}", run=run)
    rate = sum(r["resolved"] for r in rows) / len(rows)
    run.log_metrics({f"resolved_arm_{arm}": rate})
    print(arm, f"{rate:.3f}", len(rows))

# %% [markdown]
# ## 5 · Ingest

# %%
frames = [
    load_results(out_root / f"arm_{a}").with_columns(pl.lit(a).alias("arm"))
    for a in run_arms
    if (out_root / f"arm_{a}").exists()
]
frames += [df.with_columns(pl.lit(arm).alias("arm")) for arm, df in reuse.items()]
results = pl.concat(frames, how="diagonal_relaxed").with_columns(
    pl.col("agent_error").str.contains("ContextWindow").fill_null(False).alias("context_death")
)
results.write_parquet(run.dir / "results.parquet")
summary = (
    results.group_by("arm")
    .agg(
        pl.len().alias("n"),
        pl.col("resolved").mean().alias("resolved"),
        pl.col("context_death").mean().alias("context_death"),
        pl.col("submitted").mean().alias("submitted"),
        pl.col("agent_seconds").median().alias("median_s"),
        pl.col("prompt_tokens").mean().alias("mean_prompt_tokens"),
        pl.col("context_retries").fill_null(0).mean().alias("mean_retries"),
    )
    .sort("arm")
)
display(summary)

# %% [markdown]
# ## 6 · Pre-registered comparisons

# %%
rows = []
for a, b in P.comparisons:
    left = results.filter(pl.col("arm") == a).select(
        "instance_id", "seed", "repo", pl.col("resolved").alias("x")
    )
    right = results.filter(pl.col("arm") == b).select(
        "instance_id", "seed", pl.col("resolved").alias("y")
    )
    pair = left.join(right, on=["instance_id", "seed"])
    if pair.is_empty():
        continue
    d = stats.paired_diff(
        pair["x"].cast(int), pair["y"].cast(int), strata=pair["repo"], seed=cfg.seed
    )
    mc = stats.mcnemar_exact(pair["x"], pair["y"])
    rows.append(
        {
            "comparison": f"{b} - {a}",
            "rate_a": pair["x"].mean(),
            "rate_b": pair["y"].mean(),
            "diff": d.point,
            "lo": d.lo,
            "hi": d.hi,
            "n": d.n,
            "only_a": mc.only_a,
            "only_b": mc.only_b,
            "p_mcnemar": mc.p_value,
            "significant": d.lo > 0 or d.hi < 0,
        }
    )
comparisons = pl.DataFrame(rows)
run.event("comparisons", rows=comparisons.to_dicts())
display(comparisons)

# %% [markdown]
# ## 7 · Paper figure

# %%
fig, (ax1, ax2) = plotting.figure("double", aspect=0.32, ncols=2)
s = summary.sort("arm")
cis = [stats.wilson_ci(int(r["resolved"] * r["n"] + 0.5), r["n"]) for r in s.iter_rows(named=True)]
ax1.errorbar(
    s["arm"].to_list(),
    [c.point * 100 for c in cis],
    fmt="o",
    capsize=3,
    yerr=[[(c.point - c.lo) * 100 for c in cis], [(c.hi - c.point) * 100 for c in cis]],
)
ax1.set(ylabel="Resolved (%)", title="(a) Resolved rate by arm")
ax2.bar(s["arm"].to_list(), (s["context_death"] * 100).to_list(), color=plotting.PALETTE[3])
ax2.set(ylabel="Runs ending in context rejection (%)", title="(b) Context deaths")
fig.tight_layout()
suffix = f"_{P.label}" if P.label else ""
plotting.save_fig(fig, f"e1b_context_fixes{suffix}", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 8 · Teardown

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
summary.write_csv(tables / f"e1b_arms{suffix}.csv")
comparisons.write_csv(tables / f"e1b_comparisons{suffix}.csv")
run.finish()
print((run.dir / "summary.json").read_text())

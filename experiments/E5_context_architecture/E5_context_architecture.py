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
# # E5 — Phase-decomposed context architecture (C2, gate G3)
#
# **H4 / G3 (pre-registered).** A phase-decomposed agent raises the resolved rate by at
# least 5 points over the tuned monolithic agent at equal wall-clock (same 6-min budget,
# same task × seed pairs); falsified if the paired 95% CI of the gain includes 0.
#
# **Agent** (`src/build_submission.py`, declarative YAML only):
# `SequentialAgent[Localizer, Reproducer, LoopAgent×3[Editor, Verifier]]`. Every phase
# uses `include_contents: none` (fresh context: previous phase's note + own tool calls) and
# gets the issue plus structured hand-offs through session state (`output_key`). Prompts
# share a byte-identical prefix. Same model, sampling, adapters and output-reserve fix as
# the baseline (E1b arm B), so the difference isolates the architecture.

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
from src.build_submission import build
from src.params import Params

from rbcore import RunContext, load_config, plotting, stats
from swebox import sandbox as sb
from swebox.analysis import load_results, load_steps
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
display(
    Markdown(
        f"config `{cfg.hash()}` · {P.variant} · {len(tasks)} tasks × seeds {P.seeds} "
        f"= **{len(tasks) * len(P.seeds)} sessions** · baseline: `{P.baseline_dir}`"
    )
)

# %% [markdown]
# ## 2 · Preflight

# %%
import httpx

served = {
    m["id"]
    for m in httpx.get(f"{cfg.serving.base_url}/v1/models", timeout=5)
    .raise_for_status()
    .json()["data"]
}
if P.adapter:
    assert P.adapter in served, f"adapter not served: {served} (infra/vllm/serve_submission.sh)"
baseline = pl.DataFrame()
if P.baseline_dir:
    baseline = load_results(Path(P.baseline_dir)).filter(
        pl.col("instance_id").is_in(task_ids) & pl.col("seed").is_in(P.seeds)
    )
    display(Markdown(f"baseline rows: {baseline.height} / {len(task_ids) * len(P.seeds)}"))
sb.client().ping()

# %% [markdown]
# ## 3 · Open the run and build the submission
# The generated submission is stored in the run's artifacts (exactly what was evaluated).

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 9 with run.finish()
out_root = Path(P.results_dir) if P.results_dir else run.artifacts / "swebox"
run.event("results_dir", path=str(out_root), resumed=P.results_dir is not None)
submission = build(
    P.variant, run.artifacts / "submission" / P.variant, P.output_reserve, adapter=P.adapter
)
print(run.dir, submission)

# %% [markdown]
# ## 4 · Run

# %%
spec = EvalSpec(
    submission=submission,
    limits=limits,
    budget=Budget(time_minutes=P.time_minutes),
    target=ServingTarget(
        api_base=f"{cfg.serving.base_url}/v1",
        served_model=cfg.serving.model,
        thinking_translation=P.thinking_translation,
    ),
    seeds=tuple(P.seeds),
    concurrency=P.concurrency,
    tag=P.variant,
)
with run.phase("variant", variant=P.variant):
    rows = await evaluate(tasks, spec, out_root / P.variant, run=run)
run.log_metrics({"resolved": sum(r["resolved"] for r in rows) / len(rows)})

# %% [markdown]
# ## 5 · Ingest

# %%
results = load_results(out_root / P.variant).with_columns(pl.lit(P.variant).alias("arm"))
if not baseline.is_empty():
    results = pl.concat(
        [results, baseline.with_columns(pl.lit("baseline").alias("arm"))], how="diagonal_relaxed"
    )
results = results.with_columns(
    pl.col("agent_error").str.contains("ContextWindow").fill_null(False).alias("context_death")
)
results.write_parquet(run.dir / "results.parquet")
summary = (
    results.group_by("arm")
    .agg(
        pl.len().alias("n"),
        pl.col("resolved").mean(),
        pl.col("context_death").mean(),
        pl.col("submitted").mean(),
        pl.col("agent_seconds").mean().alias("mean_s"),
        pl.col("prompt_tokens").mean(),
        pl.col("model_calls").mean(),
    )
    .sort("arm")
)
display(summary)

# %% [markdown]
# ## 6 · Phase-level behaviour
# Per phase: model calls, peak prompt size, and how often each phase ran (fix-loop
# iterations, pipeline re-runs after harness nudges).

# %%
steps = load_steps(out_root / P.variant)
phase_calls = (
    steps.filter(pl.col("kind") == "model_response")
    .group_by("agent")
    .agg(
        pl.len().alias("calls"),
        pl.col("prompt_tokens").max().alias("peak_prompt"),
        pl.col("prompt_tokens").mean().alias("mean_prompt"),
        (pl.col("cached_tokens") / pl.col("prompt_tokens")).mean().alias("cache_hit"),
    )
    .sort("calls", descending=True)
)
display(phase_calls)
display(steps.filter(pl.col("kind") == "model_error").group_by("agent").len())

# %% [markdown]
# ## 7 · G3 / H4 test

# %%
g3: dict = {"verdict": "NO_BASELINE"}
if not baseline.is_empty():
    pair = (
        results.filter(pl.col("arm") == "baseline")
        .select("instance_id", "seed", "repo", pl.col("resolved").alias("b"))
        .join(
            results.filter(pl.col("arm") == P.variant).select(
                "instance_id", "seed", pl.col("resolved").alias("p")
            ),
            on=["instance_id", "seed"],
        )
    )
    d = stats.paired_diff(
        pair["b"].cast(int), pair["p"].cast(int), strata=pair["repo"], seed=cfg.seed
    )
    mc = stats.mcnemar_exact(pair["b"], pair["p"])
    if d.lo > 0 and d.point >= P.g3_min_gain:
        verdict = "PASS"
    elif d.lo <= 0 <= d.hi:
        verdict = "FAIL (gain within noise)"
    elif d.hi < 0:
        verdict = "FAIL (worse)"
    else:
        verdict = "FAIL (gain below 5 points)"
    g3 = {
        "verdict": verdict,
        "gain": d.point,
        "lo": d.lo,
        "hi": d.hi,
        "n": d.n,
        "only_baseline": mc.only_a,
        "only_phase": mc.only_b,
        "p_mcnemar": mc.p_value,
    }
    run.event("g3", **g3)
display(Markdown(f"**G3 / H4: {g3['verdict']}**  \n{g3}"))

# %% [markdown]
# ## 8 · Paper figure

# %%
fig, (ax1, ax2) = plotting.figure("double", aspect=0.32, ncols=2)
s = summary.sort("arm")
labels = [P.baseline_label if a == "baseline" else P.variant for a in s["arm"]]
cis = [stats.wilson_ci(int(r["resolved"] * r["n"] + 0.5), r["n"]) for r in s.iter_rows(named=True)]
ax1.errorbar(
    labels,
    [c.point * 100 for c in cis],
    fmt="o",
    capsize=3,
    yerr=[[(c.point - c.lo) * 100 for c in cis], [(c.hi - c.point) * 100 for c in cis]],
)
ax1.set(ylabel="Resolved (%)", title=f"(a) G3: {g3['verdict']}")
ax2.bar(
    phase_calls["agent"].to_list(), phase_calls["peak_prompt"].to_list(), color=plotting.PALETTE[1]
)
ax2.axhline(32768 - 8192, ls="--", color=plotting.PALETTE[3], lw=0.8, label="usable prompt")
ax2.set(ylabel="Peak prompt tokens", title="(b) Per-phase context size")
ax2.tick_params(axis="x", labelrotation=20)
ax2.legend(fontsize=6)
fig.tight_layout()
name = P.variant + (f"_{P.label}" if P.label else "")
plotting.save_fig(fig, f"e5_phase_agent_{name}", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 9 · Teardown

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
summary.write_csv(tables / f"e5_{name}_summary.csv")
phase_calls.write_csv(tables / f"e5_{name}_phases.csv")
pl.DataFrame([g3]).write_csv(tables / f"e5_{name}_g3.csv")
run.finish()
print((run.dir / "summary.json").read_text())

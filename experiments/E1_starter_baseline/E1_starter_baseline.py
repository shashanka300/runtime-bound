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
# # E1 — Starter baseline under realistic budgets
#
# **Question.** How does the organizers' starter agent behave on the public tasks under
# 6 and 10 minute per-task budgets, and does our local harness (`swebox`) grade correctly?
#
# **Feeds** H2 (where tokens go) and H3 (failure modes) through the step-level
# trajectories every task writes; E2 replays them.
#
# **Gates.** (a) harness self-check: gold patches resolve (≥ `min_gold_rate`), empty
# patches do not; (b) G1b: the agent runs end-to-end on the fidelity subset.
#
# **Prerequisites.** `bash infra/vllm/serve_submission.sh` (base model + the starter's
# LoRA adapters); optionally `bash infra/services.sh start`.

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
from src.params import Params

from rbcore import RunContext, load_config, plotting, stats
from rbcore.paths import data_root
from swebox import sandbox as sb
from swebox.analysis import load_results, load_steps
from swebox.evaluate import EvalSpec, evaluate, verify_only
from swebox.submission import ServingTarget
from swebox.tasks import load_tasks, select
from swebox.tools import Budget

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
tasks = select(load_tasks(), P.subset)
submission = data_root() / P.submission
limits = sb.Limits(cfg.sandbox.mem_gib or 8, cfg.sandbox.cpus or 4)
display(
    Markdown(
        f"config `{cfg.hash()}` · **{len(tasks)} tasks** × seeds {P.seeds} × budgets "
        f"{P.time_budgets_min} min · sandbox `{cfg.sandbox.profile}` ({limits.mem_gib} GiB, "
        f"{limits.cpus} vCPU) · concurrency {P.concurrency}"
    )
)
pl.DataFrame(
    [
        {"instance_id": t.instance_id, "repo": t.repo_short, "test_files": len(t.test_files)}
        for t in tasks
    ]
)

# %% [markdown]
# ## 2 · Preflight
# Docker, the sandbox image, and the served models (base + every adapter the submission
# references).

# %%
import httpx

from swebox.submission import compile_submission

dc = sb.client()
dc.ping()
sb.ensure_base_image(dc)
served = {
    m["id"]
    for m in httpx.get(f"{cfg.serving.base_url}/v1/models", timeout=5)
    .raise_for_status()
    .json()["data"]
}


async def _stub(**_: object) -> str:
    return ""


compiled = compile_submission(
    submission,
    dict.fromkeys(
        (
            "run_command",
            "read_file",
            "edit_file",
            "write_file",
            "get_status",
            "submit_patch",
            "get_code_neighbors",
            "search_similar_code",
            "get_code_subgraph",
        ),
        _stub,
    ),
    ServingTarget(),
)
missing = set(compiled.adapters) - served
display(
    Markdown(
        f"served: `{sorted(served)}` · submission adapters: `{compiled.adapters}` · "
        f"agents: {compiled.agent_count}"
    )
)
assert not missing, f"adapters not served: {missing} (use infra/vllm/serve_submission.sh)"

# %% [markdown]
# ## 3 · Open the run
# One `RunContext` for the whole experiment: telemetry at 1 Hz, each task a phase.

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 10 with run.finish()
out_root = Path(P.results_dir) if P.results_dir else run.artifacts / "swebox"
run.event("results_dir", path=str(out_root), resumed=P.results_dir is not None)
print(run.dir)

# %% [markdown]
# ## 4 · Harness self-check (no model)
# Gold (reference) patches must resolve and empty patches must not. Each task gets a
# status: `gradable`, `passes_without_fix` (empty patch passes: not discriminative),
# `infeasible:<reason>` (fails under the official rules themselves: air-gapped network,
# `-p no:anyio`) or `env_mismatch` (a harness environment defect). The gate is the gold
# rate among tasks that are feasible under the official rules.

# %%
from swebox.analysis import classify_selfcheck

gradability = pl.DataFrame()
if P.selfcheck:
    with run.phase("selfcheck"):
        gold = await verify_only(tasks, "gold", limits, out_root, concurrency=4, run=run)
        empty = await verify_only(tasks, "empty", limits, out_root, concurrency=6, run=run)
    gradability = classify_selfcheck(gold + empty, out_root)
    gradability.write_parquet(run.dir / "gradability.parquet")
    display(gradability.group_by("repo", "status").len().sort("repo", "status"))
    display(gradability.filter(pl.col("status") != "gradable"))
    feasible = gradability.filter(~pl.col("status").str.starts_with("infeasible"))
    gold_rate = float(feasible["gold"].mean())
    run.log_metrics(
        {
            "selfcheck_gold_rate_feasible": gold_rate,
            "n_gradable": float((gradability["status"] == "gradable").sum()),
        }
    )
    run.event("gate_selfcheck", gold_rate_feasible=gold_rate, passed=gold_rate >= P.min_gold_rate)
    assert gold_rate >= P.min_gold_rate, "harness gate failed; fix the environment first"

# %% [markdown]
# ## 5 · Run the agent
# Each budget is a separate condition (`tag`) with its own results directory. Re-running
# this cell resumes: finished (task, seed) pairs are skipped.

# %%
for minutes in P.time_budgets_min:
    tag = f"t{minutes:g}min"
    spec = EvalSpec(
        submission=submission,
        limits=limits,
        budget=Budget(time_minutes=minutes, tool_calls=P.max_tool_calls),
        target=ServingTarget(api_base=f"{cfg.serving.base_url}/v1", served_model=cfg.serving.model),
        seeds=tuple(P.seeds),
        concurrency=P.concurrency,
        tag=tag,
    )
    with run.phase("condition", tag=tag):
        rows = await evaluate(tasks, spec, out_root / tag, run=run)
    rate = sum(r["resolved"] for r in rows) / len(rows)
    run.log_metrics({f"resolved_rate_{tag}": rate})
    print(tag, f"{rate:.3f}", len(rows))

# %% [markdown]
# ## 6 · Live hardware view (re-run any time)

# %%
gpu = run.hw("gpu")
fig, axes = plotting.figure("double", aspect=0.3, ncols=3)
for ax, col, label in zip(
    axes,
    ["power_w", "util_gpu_pct", "mem_used_mib"],
    ["GPU power (W)", "GPU util (%)", "GPU mem (MiB)"],
    strict=True,
):
    ax.plot(gpu["time"], gpu[col], lw=0.5)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", labelrotation=30)
fig.tight_layout()

# %% [markdown]
# ## 7 · Ingest

# %%
dirs = [out_root / f"t{m:g}min" for m in P.time_budgets_min]
results = load_results(*dirs)
steps = load_steps(*dirs)
results.write_parquet(run.dir / "results.parquet")
steps.write_parquet(run.dir / "steps.parquet")
print(results.shape, steps.shape)
results.select(
    "tag",
    "instance_id",
    "seed",
    "resolved",
    "end_reason",
    "agent_seconds",
    "turns",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "thought_tokens",
)

# %% [markdown]
# ## 8 · Analysis
# Resolved rate per budget with Wilson 95% CIs; paired comparison of the two budgets on
# the same (task, seed) pairs (bootstrap CI of the difference, exact McNemar); end reasons
# and token volume.

# %%
per_tag_rows = []
for (tag,), g in results.group_by("tag", maintain_order=True):
    est = stats.wilson_ci(int(g["resolved"].sum()), g.height)
    per_tag_rows.append(
        {
            "tag": tag,
            "n": g.height,
            "rate": est.point,
            "lo": est.lo,
            "hi": est.hi,
            "median_agent_s": g["agent_seconds"].median(),
            "mean_turns": g["turns"].mean(),
        }
    )
TAG_ORDER = pl.Enum([f"t{m:g}min" for m in P.time_budgets_min])  # budgets in numeric order
per_tag = pl.DataFrame(per_tag_rows).sort(pl.col("tag").cast(TAG_ORDER))
display(per_tag)

if len(P.time_budgets_min) == 2:
    a_tag, b_tag = (f"t{m:g}min" for m in P.time_budgets_min)
    pair = (
        results.filter(pl.col("tag") == a_tag)
        .select("instance_id", "seed", "repo", pl.col("resolved").alias("a"))
        .join(
            results.filter(pl.col("tag") == b_tag).select(
                "instance_id", "seed", pl.col("resolved").alias("b")
            ),
            on=["instance_id", "seed"],
        )
    )
    diff = stats.paired_diff(pair["a"].cast(int), pair["b"].cast(int), strata=pair["repo"])
    mc = stats.mcnemar_exact(pair["a"], pair["b"])
    display(
        Markdown(
            f"**{b_tag} − {a_tag}:** {diff} · McNemar only-a={mc.only_a}, "
            f"only-b={mc.only_b}, p={mc.p_value:.3f}"
        )
    )

display(results.group_by("tag", "end_reason").len().sort("tag", "len", descending=[False, True]))
tokens = (
    results.group_by("tag")
    .agg(
        pl.col("prompt_tokens").sum(),
        pl.col("cached_tokens").sum(),
        pl.col("completion_tokens").sum(),
        pl.col("thought_tokens").sum(),
        pl.col("compactions").sum(),
        pl.col("model_calls").sum(),
    )
    .sort(pl.col("tag").cast(TAG_ORDER))
)
display(tokens)

# %% [markdown]
# ## 9 · Paper figures

# %%
fig, (ax1, ax2) = plotting.figure("double", aspect=0.32, ncols=2)
xs = list(range(per_tag.height))
ax1.errorbar(
    xs,
    per_tag["rate"],
    fmt="o",
    capsize=3,
    yerr=[(per_tag["rate"] - per_tag["lo"]).to_list(), (per_tag["hi"] - per_tag["rate"]).to_list()],
)
ax1.set_xticks(xs, per_tag["tag"].to_list())
ax1.set(ylabel="Resolved rate", title="(a) Starter agent vs per-task budget", ylim=(0, 1))
labels = tokens["tag"].to_list()
uncached = (tokens["prompt_tokens"] - tokens["cached_tokens"]) / 1e6
stack = [
    ("prompt, cache hit", tokens["cached_tokens"] / 1e6),
    ("prompt, recomputed", uncached),
    ("thinking", tokens["thought_tokens"] / 1e6),
    ("output", (tokens["completion_tokens"] - tokens["thought_tokens"]).clip(0) / 1e6),
]
bottom = [0.0] * len(labels)
for label, vals in stack:
    ax2.bar(labels, vals.to_list(), bottom=bottom, label=label)
    bottom = [b + v for b, v in zip(bottom, vals.to_list(), strict=True)]
ax2.set(ylabel="Tokens (M)", title="(b) Token volume by kind")
ax2.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.tight_layout()
plotting.save_fig(fig, "e1_resolved_and_tokens", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 10 · Teardown

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
per_tag.write_csv(tables / "e1_resolved_by_budget.csv")
if not gradability.is_empty():
    gradability.write_csv(tables / "e1_task_gradability.csv")
run.finish()
print((run.dir / "summary.json").read_text())

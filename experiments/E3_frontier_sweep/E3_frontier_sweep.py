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
# # E3 — How should 31B spend its runtime?
#
# **Scope (design change 2026-09-26, see README).** 31B-int4 only: the goal is improving the
# 31B agent, and E2/H3 showed failures are context exhaustion, not model capacity, so the
# planned 12B comparison (H1) was dropped.
#
# **Design.** thinking budget {2k, 4k, 8k} × attempts {1, 2, 4} on `dev40`. An attempt is an independent fresh-context session; 4 are run per condition and
# task, and k attempts are the first k (`src/frontier.py`). Same agent everywhere (starter
# prompts, adapter-free, `max_output_tokens = thinking + 4096`; `src/variants.py`).
# The vLLM server is (re)started automatically, so the sweep runs unattended.

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
from src import frontier as fr
from src.params import Params
from src.variants import make_variant

from rbcore import RunContext, load_config, plotting
from rbcore.paths import data_root
from swebox import sandbox as sb
from swebox.analysis import load_results
from swebox.evaluate import EvalSpec, evaluate
from swebox.serving import ModelSpec, ensure_model
from swebox.submission import ServingTarget
from swebox.tasks import load_tasks, select
from swebox.tools import Budget

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
tasks = select(load_tasks(), P.subset)
limits = sb.Limits(cfg.sandbox.mem_gib or 8, cfg.sandbox.cpus or 4)
conditions = [(m, t) for m in P.models for t in P.thinking_budgets]
n_runs = len(conditions) * len(tasks) * P.attempts
display(
    Markdown(
        f"config `{cfg.hash()}` · {len(tasks)} tasks × {len(conditions)} conditions × "
        f"{P.attempts} attempts = **{n_runs} sessions** (~{n_runs * 2.6 / 2 / 60:.0f} h at E1's "
        f"pace, 2 concurrent)"
    )
)
pl.DataFrame([{"model": m.key, "thinking": t, "tag": f"{m.key}_t{t}"} for m, t in conditions])

# %% [markdown]
# ## 2 · Preflight
# Weights for every model must be in the HF cache; Docker and the sandbox image ready.

# %%
hub = data_root() / "hf" / "hub"
weights = {m.key: (hub / f"models--{m.hf_id.replace('/', '--')}").is_dir() for m in P.models}
display(weights)
assert all(weights.values()), f"missing weights: {weights} (infra/data/download.sh)"
dc = sb.client()
dc.ping()
sb.ensure_base_image(dc)

# %% [markdown]
# ## 3 · Open the run

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 9 with run.finish()
out_root = Path(P.results_dir) if P.results_dir else run.artifacts / "swebox"
run.event("results_dir", path=str(out_root), resumed=P.results_dir is not None)
print(run.dir)

# %% [markdown]
# ## 4 · Sweep
# Grouped by model (one fits on the GPU at a time). Re-running resumes: finished
# (task, attempt) results are skipped per condition.

# %%
from swebox.serving import stop_server

base = data_root() / P.base_submission
if P.fresh_server and not P.results_dir:
    stop_server()
for m in P.models:
    with run.phase("serve_switch", model=m.key):
        ensure_model(ModelSpec(m.hf_id, m.served_name, tuple(m.extra_args)), cfg.serving.base_url)
    for thinking in P.thinking_budgets:
        tag = f"{m.key}_t{thinking}"
        sub = make_variant(
            base, run.artifacts / "submissions" / tag, m.alias, thinking, P.output_reserve
        )
        spec = EvalSpec(
            submission=sub,
            limits=limits,
            budget=Budget(time_minutes=P.time_minutes),
            target=ServingTarget(api_base=f"{cfg.serving.base_url}/v1", served_model=m.served_name),
            seeds=tuple(range(P.attempts)),
            concurrency=m.concurrency,
            tag=tag,
        )
        with run.phase("condition", tag=tag, model=m.key, thinking=thinking):
            rows = await evaluate(tasks, spec, out_root / tag, run=run)
        rate = sum(r["resolved"] for r in rows) / len(rows)
        run.log_metrics({f"resolved_{tag}": rate})
        print(tag, f"{rate:.3f}", len(rows))

# Leave the default model served for the next experiment.
ensure_model(
    ModelSpec(P.models[0].hf_id, P.models[0].served_name, tuple(P.models[0].extra_args)),
    cfg.serving.base_url,
)

# %% [markdown]
# ## 5 · Live hardware view (re-run any time)

# %%
gpu = run.hw("gpu")
fig, ax = plotting.figure("double", aspect=0.25)
ax.plot(gpu["time"], gpu["power_w"], lw=0.4)
ax.set_ylabel("GPU power (W)")
ev = run.events()
for w in (
    ev.filter(pl.col("event") == "phase_start")
    .filter(pl.col("phase") == "condition")
    .iter_rows(named=True)
):
    ax.axvline(pl.from_epoch(pl.Series([w["t_ns"]]), "ns")[0], color="grey", lw=0.5)

# %% [markdown]
# ## 6 · Ingest

# %%
dirs = [out_root / f"{m.key}_t{t}" for m, t in conditions]
results = load_results(*[d for d in dirs if d.exists()]).with_columns(
    pl.col("tag").str.extract(r"^(.+)_t\d+$").alias("model"),
    pl.col("tag").str.extract(r"_t(\d+)$").cast(pl.Int64).alias("thinking"),
)
results.write_parquet(run.dir / "results.parquet")
display(
    results.group_by("model", "thinking")
    .agg(
        pl.len().alias("sessions"),
        pl.col("resolved").mean().alias("per_session_rate"),
        pl.col("agent_seconds").median().alias("median_s"),
        pl.col("agent_error").str.contains("ContextWindow").mean().alias("context_death"),
    )
    .sort("model", "thinking")
)

# %% [markdown]
# ## 7 · Frontier and pre-registered comparisons
# **Q-A (fresh-context restarts):** best k>1 configuration vs best k=1 configuration.
# **Q-B (thinking budget):** 8k vs 2k thinking, single attempt.
# **Recommendation:** best deploy configuration with mean seconds <= `recommend_budget_s`.

# %%
attempts = fr.attempts_table(results)
task_scores = fr.per_task(attempts, P.ks)
configs = fr.configs(task_scores, seed=cfg.seed)
front = fr.pareto(configs)
task_scores.write_parquet(run.dir / "task_scores.parquet")
display(configs)

model = P.models[0].key
single = configs.filter(pl.col("k") == 1).sort("deploy", descending=True).row(0, named=True)
multi = configs.filter(pl.col("k") > 1).sort("deploy", descending=True).row(0, named=True)
tb = sorted(P.thinking_budgets)
comparisons = pl.DataFrame(
    [
        {
            "question": "Q-A fresh-context restarts",
            **fr.paired(task_scores, single, multi, seed=cfg.seed),
        },
        {
            "question": "Q-B thinking budget",
            **fr.paired(
                task_scores,
                {"model": model, "thinking": tb[0], "k": 1},
                {"model": model, "thinking": tb[-1], "k": 1},
                seed=cfg.seed,
            ),
        },
    ]
)
fits = configs.filter(pl.col("deploy_seconds") <= P.recommend_budget_s)
recommended = fits.sort("deploy", descending=True).row(0, named=True) if fits.height else None
run.event("comparisons", rows=comparisons.to_dicts(), recommended=recommended)
display(comparisons)
display(
    Markdown(
        "**Recommended within budget:** "
        + (
            f"thinking {recommended['thinking']}, k={recommended['k']}: deploy "
            f"{recommended['deploy']:.1%} [{recommended['deploy_lo']:.1%}, "
            f"{recommended['deploy_hi']:.1%}] at {recommended['deploy_seconds']:.0f} s/task"
            if recommended
            else "none fits"
        )
    )
)

# %% [markdown]
# ## 8 · Paper figures
# x = mean agent seconds per task on the 5090 (the eval box is slower; E0 gives the scale);
# the dotted line marks 360 s, i.e. 12 h over 120 tasks.

# %%
fig, ax = plotting.figure("single", aspect=0.8)
markers = {2048: "o", 4096: "s", 8192: "^"}
for i, m in enumerate(P.models):
    c = configs.filter(pl.col("model") == m.key)
    for r in c.iter_rows(named=True):
        ax.scatter(
            r["deploy_seconds"],
            r["deploy"] * 100,
            marker=markers.get(r["thinking"], "o"),
            s=10 + 12 * r["k"],
            color=plotting.PALETTE[i],
            alpha=0.85,
        )
        ax.scatter(
            r["oracle_seconds"],
            r["oracle"] * 100,
            marker=markers.get(r["thinking"], "o"),
            s=10 + 12 * r["k"],
            facecolors="none",
            edgecolors=plotting.PALETTE[i],
            lw=0.6,
        )
    ax.scatter([], [], color=plotting.PALETTE[i], label=m.key)
if front.height:
    ax.plot(front["deploy_seconds"], front["deploy"] * 100, color="k", lw=0.8, label="Pareto")
for t, mk in markers.items():
    ax.scatter([], [], marker=mk, color="grey", label=f"thinking {t // 1024}k")
ax.axvline(360, ls=":", color="grey", lw=0.8)
ax.set(
    xlabel="Mean wall-clock per task (s, local)",
    ylabel="Resolved (%)",
    title="31B frontier (filled: deploy, hollow: pass@k)",
)
ax.legend(fontsize=5, loc="lower right")
plotting.save_fig(fig, "e3_frontier", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 9 · Teardown

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
configs.write_csv(tables / "e3_configs.csv")
front.write_csv(tables / "e3_pareto.csv")
comparisons.write_csv(tables / "e3_comparisons.csv")
run.finish()
print((run.dir / "summary.json").read_text())

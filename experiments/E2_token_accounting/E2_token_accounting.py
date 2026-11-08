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
# # E2 — Where do the tokens go? (H2)
#
# **H2 (pre-registered).** Over 40% of tokens are re-reads, re-prefill, or truncated tool
# output. Falsified if the waste share is under 20%.
#
# **Method.** Replay the step-level traces that every swebox task writes (no GPU). Two views
# per model call (see `src/accounting.py`):
# - *compute*: prompt = cached + new + **recomputed** prefill; decode = thinking + output;
# - *occupancy*: what fills the prompt, with tool outputs tracked as segments that stay in
#   context and flagged **re-read**, **truncated** or oversized.
#
# Waste = re-read + truncated segment tokens + recomputed prefill, as a share of all
# processed tokens, per task. The categories can overlap, so the sum is an upper bound and
# the largest category a lower bound; both are tested.
#
# **Decision rule.** Supported if the lower bound's 95% CI lies above 40%; falsified if the
# upper bound's CI lies below 20%; otherwise inconclusive.

# %% [markdown]
# ## 0 · Parameters

# %% tags=["parameters"]
CONFIG_OVERRIDES: dict = {}

# %% [markdown]
# ## 1 · Setup & provenance

# %%
from pathlib import Path

import numpy as np
import polars as pl
from IPython.display import Markdown, display
from src import accounting as acc
from src.params import Params

from rbcore import RunContext, load_config, plotting, stats
from rbcore.paths import data_root
from swebox.analysis import load_results, load_steps

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
SUFFIX = f"_{P.label}" if P.label else ""  # keeps outputs of different inputs apart
plotting.use_style()
display(Markdown(f"config `{cfg.hash()}` · inputs: `{P.input_dirs or 'self-produced'}`"))

# %% [markdown]
# ## 2 · Open the run and resolve inputs
# With no `input_dirs`, a small swebox run is produced first (needs the vLLM server with the
# submission's adapters), so E2 never depends on another experiment having run.

# %%
run = RunContext(cfg, exp_dir=EXP_DIR).start()  # closed in section 8 with run.finish()

if P.input_dirs:
    inputs = [Path(d) for d in P.input_dirs]
else:
    from swebox import sandbox as sb
    from swebox.evaluate import EvalSpec, evaluate
    from swebox.submission import ServingTarget
    from swebox.tasks import load_tasks, select
    from swebox.tools import Budget

    spec = EvalSpec(
        submission=data_root() / P.fallback_submission,
        limits=sb.Limits(cfg.sandbox.mem_gib or 8, cfg.sandbox.cpus or 4),
        budget=Budget(time_minutes=P.fallback_minutes),
        target=ServingTarget(api_base=f"{cfg.serving.base_url}/v1", served_model=cfg.serving.model),
        tag=f"t{P.fallback_minutes:g}min",
    )
    out = run.artifacts / "fallback" / spec.tag
    with run.phase("fallback_eval"):
        await evaluate(select(load_tasks(), P.fallback_subset), spec, out, run=run)
    inputs = [out]

missing = [d for d in inputs if not (d / "traces").is_dir()]
assert not missing, f"no traces/ under {missing}"
run.event("inputs", dirs=[str(d) for d in inputs])
[str(d) for d in inputs]

# %% [markdown]
# ## 3 · Load traces and calibrate characters → tokens
# Request contents are logged as character counts by kind; a per-agent linear fit maps them
# to tokens (the intercept absorbs tool schemas and the chat template).

# %%
with run.phase("load"):
    steps = load_steps(*inputs)
    results = load_results(*inputs)
    calls = acc.model_calls(steps)
    fits = acc.fit_chars_to_tokens(calls)
fit_table = pl.DataFrame([f.__dict__ for f in fits.values()])
display(fit_table)
display(
    Markdown(
        f"{results.height} task runs · {calls.height} model calls · "
        f"{int(calls['overflow'].sum())} rejected for exceeding the 32k window"
    )
)

fig, ax = plotting.figure("single")
ok = calls.filter(pl.col("prompt_tokens").is_not_null())
for agent, g in ok.group_by("agent"):
    ax.scatter(g["chars"] / 1e3, g["prompt_tokens"] / 1e3, s=2, alpha=0.4, label=agent[0])
    f = fits[agent[0]]
    xs = np.linspace(0, float(g["chars"].max()), 50)
    ax.plot(xs / 1e3, (f.tokens_per_char * xs + f.intercept) / 1e3, lw=0.8)
ax.set(
    xlabel="Request characters (k)",
    ylabel="Prompt tokens (k)",
    title="Characters → tokens calibration",
)
ax.legend(fontsize=6)
plotting.save_fig(fig, f"e2_char_token_fit{SUFFIX}", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 4 · Compute view: what the GPU processed

# %%
cv = acc.compute_view(calls)
compute_totals = (
    cv.select(
        pl.col("cached_tokens").sum().alias("prompt: cache hit"),
        pl.col("new").sum().alias("prompt: new"),
        pl.col("recomputed").sum().alias("prompt: recomputed"),
        pl.col("thought_tokens").sum().alias("decode: thinking"),
        pl.col("output_tokens").sum().alias("decode: output"),
    )
    .unpivot(variable_name="category", value_name="tokens")
    .with_columns((pl.col("tokens") / pl.col("tokens").sum()).alias("share"))
)
display(compute_totals)

# %% [markdown]
# ## 5 · Occupancy view: what fills the prompt
# Tool outputs are segments that stay in context for every later call of that context.

# %%
seg = acc.segments(steps, calls, fits, P.reread_overlap, P.oversized_chars)
display(
    seg.group_by("tool")
    .agg(
        pl.len().alias("n"),
        pl.col("reread").sum(),
        pl.col("truncated").sum(),
        pl.col("oversized").sum(),
        pl.col("chars").median().alias("median_chars"),
        pl.col("lifetime").mean().round(1).alias("mean_lifetime"),
    )
    .sort("n", descending=True)
)

rate = pl.col("agent").replace_strict({a: f.tokens_per_char for a, f in fits.items()}, default=0.25)
occ = calls.filter(~pl.col("overflow")).select(
    ((pl.col("system_chars") + pl.col("user_text")) * rate).sum().alias("instructions + task"),
    (pl.col("model_text") * rate).sum().alias("model text"),
    (pl.col("thought") * rate).sum().alias("stale thinking"),
    (pl.col("tool_call") * rate).sum().alias("tool-call args"),
    (pl.col("tool_output") * rate).sum().alias("tool output"),
)
tool_total = float(occ["tool output"][0])
flag_totals = {
    "re-read": float(seg.filter("reread")["token_steps"].sum()),
    "truncated": float(seg.filter(pl.col("truncated") & ~pl.col("reread"))["token_steps"].sum()),
    "oversized": float(
        seg.filter(pl.col("oversized") & ~pl.col("truncated") & ~pl.col("reread"))[
            "token_steps"
        ].sum()
    ),
}
occupancy = pl.DataFrame(
    {
        "segment": [
            "instructions + task",
            "model text",
            "stale thinking",
            "tool-call args",
            "tool output: fresh",
            "tool output: re-read",
            "tool output: truncated",
            "tool output: oversized",
        ],
        "tokens": [
            float(occ[c][0])
            for c in ("instructions + task", "model text", "stale thinking", "tool-call args")
        ]
        + [
            max(0.0, tool_total - sum(flag_totals.values())),
            flag_totals["re-read"],
            flag_totals["truncated"],
            flag_totals["oversized"],
        ],
    }
).with_columns((pl.col("tokens") / pl.col("tokens").sum()).alias("share"))
display(occupancy)

# %% [markdown]
# ## 6 · H2 test
# Per-task waste shares; 95% CIs from a bootstrap over tasks stratified by repo.

# %%
tasks = acc.per_task(cv, seg, calls, fits).with_columns(
    pl.col("instance_id").str.replace(r"_\d+$", "").alias("repo")
)
tasks = tasks.join(results.select(*acc.KEY, "resolved", "end_reason"), on=acc.KEY, how="left")
tasks.write_parquet(run.dir / "per_task.parquet")

upper = stats.bootstrap_ci(tasks["waste_share_upper"], strata=tasks["repo"], seed=cfg.seed)
lower = stats.bootstrap_ci(tasks["waste_share_lower"], strata=tasks["repo"], seed=cfg.seed)
parts = {
    c: stats.bootstrap_ci(tasks[c], strata=tasks["repo"], seed=cfg.seed)
    for c in ("reread_share", "truncated_share", "recomputed_share")
}
if lower.lo > P.h2_supported_above:
    verdict = "SUPPORTED"
elif upper.hi < P.h2_falsified_below:
    verdict = "FALSIFIED"
else:
    verdict = "INCONCLUSIVE"
h2 = {
    "verdict": verdict,
    "n_tasks": tasks.height,
    "waste_upper": str(upper),
    "waste_lower": str(lower),
    **{k: str(v) for k, v in parts.items()},
}
run.event("h2", **h2)
run.log_metrics(
    {
        "waste_upper": upper.point,
        "waste_lower": lower.point,
        **{k: v.point for k, v in parts.items()},
    }
)
display(
    Markdown(
        f"**H2: {verdict}** (n={tasks.height} task runs)  \n"
        f"waste share, upper bound (sum): {upper}  \nlower bound (largest category): {lower}  \n"
        + "  \n".join(f"{k}: {v}" for k, v in parts.items())
    )
)
display(
    tasks.group_by("tag").agg(
        pl.len().alias("n"),
        pl.col("waste_share_upper").mean(),
        pl.col("waste_share_lower").mean(),
        pl.col("cache_hit_rate").mean(),
        pl.col("overflow_calls").sum(),
    )
)

# %% [markdown]
# ## 6b · H3: failure taxonomy (gate G2)
# One primary label per unresolved run (precedence = proximate cause, `src/taxonomy.py`),
# a patch-only label that ignores how the session ended, and behavioural markers.
# **Supported** if `context_exhausted` ranks in the top `h3_top_k` failure modes;
# **falsified** if mislocalization + wrong edits exceed 50% of failures and it does not.

# %%
from src import taxonomy as tx

from swebox.tasks import load_tasks

task_map = {t.instance_id: t for t in load_tasks()}
labels = tx.label_runs(results, inputs, task_map)
beh = (
    tx.behaviour(steps, tx.gold_source_files(task_map))
    .join(
        seg.group_by(acc.KEY).agg(pl.col("reread").sum().alias("rereads")), on=acc.KEY, how="left"
    )
    .with_columns(pl.col("rereads").fill_null(0))
)
runs = labels.join(beh, on=acc.KEY, how="left").with_columns(
    pl.col("instance_id").str.replace(r"_\d+$", "").alias("repo")
)
if P.gradability_csv:
    grad = pl.read_csv(P.gradability_csv).select("instance_id", "status")
    runs = runs.join(grad, on="instance_id", how="left").filter(pl.col("status") == "gradable")
runs.write_parquet(run.dir / "failure_labels.parquet")

failed = runs.filter(pl.col("primary") != "resolved")
modes = []
for m in tx.PRIMARY:
    est = stats.bootstrap_ci(
        (failed["primary"] == m).cast(int), strata=failed["repo"], seed=cfg.seed
    )
    modes.append(
        {
            "mode": m,
            "n": int((failed["primary"] == m).sum()),
            "share": est.point,
            "lo": est.lo,
            "hi": est.hi,
        }
    )
modes = pl.DataFrame(modes).sort("share", descending=True).with_row_index("rank", offset=1)
display(modes)
rank_ctx = int(modes.filter(pl.col("mode") == "context_exhausted")["rank"][0])
wrong = float(
    modes.filter(pl.col("mode").is_in(["mislocalized_file", "mislocalized_region", "wrong_edit"]))[
        "share"
    ].sum()
)
if modes.filter(pl.col("mode") == "context_exhausted")["n"][0] > 0 and rank_ctx <= P.h3_top_k:
    h3_verdict = "SUPPORTED"
elif wrong > 0.5:
    h3_verdict = "FALSIFIED"
else:
    h3_verdict = "INCONCLUSIVE"
h3 = {
    "verdict": h3_verdict,
    "n_runs": runs.height,
    "n_failed": failed.height,
    "context_exhausted_rank": rank_ctx,
    "localization_or_edit_share": wrong,
}
run.event("h3", **h3)
run.log_metrics({"h3_context_rank": rank_ctx, "h3_localization_or_edit_share": wrong})

# What had the runs that context killed already achieved?
cross = (
    failed.group_by("primary", "patch_only").len().sort("primary", "len", descending=[False, True])
)
ctx_runs = failed.filter(pl.col("primary") == "context_exhausted")
markers = failed.select(
    (pl.col("repeated_actions") >= 2).mean().alias("repeated_actions>=2"),
    (pl.col("rereads") >= 1).mean().alias("rereads>=1"),
    (pl.col("reverted_edits") >= 1).mean().alias("reverted_edit>=1"),
    pl.col("viewed_gold_file").mean().alias("viewed_a_gold_file"),
    (pl.col("scratch_files") >= 1).mean().alias("scratch_file_in_patch"),
)
display(
    Markdown(
        f"**H3: {h3_verdict}**: context_exhausted ranks #{rank_ctx} of {len(tx.PRIMARY)} among "
        f"{failed.height} failed runs; mislocalization + wrong edits = {wrong:.1%}.  \n"
        f"Of context-exhausted runs, {(ctx_runs['patch_only'] == 'right_region').mean():.1%} had "
        f"already edited the reference region and {ctx_runs['viewed_gold_file'].mean():.1%} had "
        f"viewed a reference file."
    )
)
display(cross)
display(markers)

# Spot-check sample for human verification of the rules.
spot = (
    failed.sample(n=min(P.spot_check_n, failed.height), seed=cfg.seed)
    .join(
        results.select(*acc.KEY, "end_reason", "agent_error", "patch_files", "test_exit_code"),
        on=acc.KEY,
        how="left",
    )
    .with_columns(pl.col("agent_error").str.slice(0, 80))
)
display(spot)

fig, (ax1, ax2) = plotting.figure("double", aspect=0.32, ncols=2)
m = modes.filter(pl.col("n") > 0).sort("share")
ax1.barh(
    m["mode"].to_list(),
    (m["share"] * 100).to_list(),
    xerr=[((m["share"] - m["lo"]) * 100).to_list(), ((m["hi"] - m["share"]) * 100).to_list()],
    color=[
        plotting.PALETTE[3] if x == "context_exhausted" else plotting.PALETTE[0] for x in m["mode"]
    ],
)
ax1.set(xlabel="% of failed runs", title=f"(a) Primary failure mode (H3: {h3_verdict})")
pc = ctx_runs.group_by("patch_only").len().sort("len", descending=True)
ax2.bar(pc["patch_only"].to_list(), pc["len"].to_list(), color=plotting.PALETTE[1])
ax2.set(ylabel="Runs", title="(b) Context-exhausted runs: patch at the moment of death")
ax2.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
plotting.save_fig(fig, f"e2_failure_taxonomy{SUFFIX}", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 7 · Paper figures

# %%
fig, axes = plotting.figure("double", aspect=0.36, ncols=3)
# (a) compute view
ax = axes[0]
left = 0.0
for row in compute_totals.iter_rows(named=True):
    ax.barh([0], [row["share"] * 100], left=[left], label=row["category"])
    left += row["share"] * 100
ax.set(xlim=(0, 100), yticks=[], xlabel="% of processed tokens", title="(a) Compute")
ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
# (b) occupancy view
ax = axes[1]
left = 0.0
for row in occupancy.iter_rows(named=True):
    ax.barh([0], [row["share"] * 100], left=[left], label=row["segment"])
    left += row["share"] * 100
ax.set(xlim=(0, 100), yticks=[], xlabel="% of prompt tokens", title="(b) Context occupancy")
ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
# (c) per-task waste distribution against the pre-registered thresholds
ax = axes[2]
ax.hist(tasks["waste_share_upper"] * 100, bins=20, alpha=0.6, label="upper (sum)")
ax.hist(tasks["waste_share_lower"] * 100, bins=20, alpha=0.6, label="lower (max)")
for thr, style in ((P.h2_falsified_below, ":"), (P.h2_supported_above, "--")):
    ax.axvline(thr * 100, color="k", ls=style, lw=0.8)
ax.set(xlabel="Waste share per task (%)", ylabel="Task runs", title=f"(c) H2: {verdict}")
ax.legend(fontsize=6)
fig.tight_layout()
plotting.save_fig(fig, f"e2_token_accounting{SUFFIX}", exp_dir=EXP_DIR, run=run)

# %%
# Context growth: prompt size per call for every root-agent context, 32k limit, overflows.
fig, ax = plotting.figure("single", aspect=0.7)
root_agent = calls.group_by("agent").len().sort("len", descending=True)["agent"][0]
rc = calls.filter(pl.col("agent") == root_agent)
est = rc.with_columns(
    pl.coalesce(
        "prompt_tokens",
        pl.col("chars") * fits[root_agent].tokens_per_char + fits[root_agent].intercept,
    ).alias("p")
)
for _, g in est.group_by(acc.CTX):
    ax.plot(g["call_idx"], g["p"] / 1e3, lw=0.3, alpha=0.25, color=plotting.PALETTE[0])
of = est.filter("overflow")
ax.scatter(
    of["call_idx"],
    of["p"] / 1e3,
    s=4,
    color=plotting.PALETTE[3],
    label="rejected: prompt + max_output > window",
)
med = (
    est.filter(~pl.col("overflow")).group_by("call_idx").agg(pl.col("p").median()).sort("call_idx")
)
ax.plot(med["call_idx"], med["p"] / 1e3, lw=1.2, color="k", label="median")
window = cfg.serving.max_model_len
ax.axhline(window / 1e3, ls="--", lw=0.8, color="grey", label=f"window {window // 1024}k")
usable = window - P.reserved_output_tokens
ax.axhline(
    usable / 1e3,
    ls="--",
    lw=0.8,
    color=plotting.PALETTE[3],
    label=f"usable prompt ({window // 1024}k - max_output {P.reserved_output_tokens // 1024}k)",
)
ax.set(xlabel="Model call index", ylabel="Prompt tokens (k)", title="Context growth")
ax.legend(fontsize=6)
plotting.save_fig(fig, f"e2_context_growth{SUFFIX}", exp_dir=EXP_DIR, run=run)

# %% [markdown]
# ## 8 · Teardown

# %%
tables = EXP_DIR / "tables"
tables.mkdir(exist_ok=True)
compute_totals.write_csv(tables / f"e2_compute_view{SUFFIX}.csv")
occupancy.write_csv(tables / f"e2_occupancy_view{SUFFIX}.csv")
fit_table.write_csv(tables / f"e2_char_token_fit{SUFFIX}.csv")
pl.DataFrame([h2]).write_csv(tables / f"e2_h2_test{SUFFIX}.csv")
pl.DataFrame([h3]).write_csv(tables / f"e2_h3_test{SUFFIX}.csv")
modes.write_csv(tables / f"e2_failure_modes{SUFFIX}.csv")
cross.write_csv(tables / f"e2_failure_crosstab{SUFFIX}.csv")
spot.write_csv(tables / f"e2_failure_spotcheck{SUFFIX}.csv")
run.finish()
print((run.dir / "summary.json").read_text())

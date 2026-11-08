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
# # Most FastAPI and requests tasks cannot pass in the offline grading environment
#
# Evidence notebook for a forum report to the organisers (draft below; **not posted**).
# Everything here is produced by the organisers' own code from the official wheelhouse
# (`swegemma 0.2.7`, `adk_submission 0.2.11`) with the competition's `wheels/` directory and
# the `swebench-sandbox` image, network off. No model is involved: each task is graded with
# its **reference (gold) patch** and with **no patch**.
#
# ---
# ## Draft forum post
#
# **Title:** Gold patches fail offline for most FastAPI and requests tasks (missing test
# dependencies in the grading environment?)
#
# Hi @Ryan Holbrook, and thanks for the wheelhouse, which made this check possible.
#
# We graded every public task's **reference patch** with the official Phase-2 code
# (`swegemma.harness.verification.verify_task`, called as `Evaluator.evaluate_task` does:
# `resolve_task_snapshot_paths`, `wheels_dir=<competition>/wheels`, image
# `swebench-sandbox:latest`, network off). A task counts as gradable when the gold patch
# resolves and the empty patch does not.
#
# | repo | gradable with the gold patch |
# |---|---|
# | Textualize/rich | 39 / 48 |
# | encode/httpx | 1 / 1 |
# | fastapi/fastapi | **1 / 67** |
# | psf/requests | **0 / 13** |
#
# The FastAPI failures are test-collection errors, e.g. `ModuleNotFoundError: No module named
# 'dirty_equals'` (`tests/test_union_body_discriminator.py`), and in the sample-verification
# path also `starlette`/`anyio` imports; the requests failures are fixture errors (187 errors on
# `psf/requests#6589` with the gold patch applied), consistent with a missing pytest plugin.
# If the scorer uses the same environment for the hidden set, every FastAPI/requests task
# scores 0 for every team, whatever the patch.
#
# Questions:
# 1. Does the scoring run install test-only dependencies (`dirty_equals`, the requests test
#    fixtures, ...) from somewhere other than the public `wheels/` directory?
# 2. Does the scorer use the `docker` or the `subprocess` sandbox (and which image)?
# 3. Were the hidden FastAPI/requests tasks validated in that same environment?
#
# A notebook reproducing this (a few minutes on CPU) is attached; with `SANDBOX = "subprocess"`
# it runs the same check inside a Kaggle notebook image.
#
# ---

# %% tags=["parameters"]
SANDBOX = "docker"  # "subprocess" to reproduce inside a Kaggle notebook image
RUN_REPRO = True  # re-grade a few tasks live in section 4
REPRO_IDS = "fastapi_12942,requests_6589,rich_3938"

# %% [markdown]
# ## 1 · Setup

# %%
import json
import subprocess
from pathlib import Path

import polars as pl
from IPython.display import Markdown, display

from rbcore import plotting

ROOT = Path.cwd().resolve()
while not (ROOT / "leaderboard").exists():
    ROOT = ROOT.parent
REPORT = ROOT / "reports" / "gradability_audit"
plotting.use_style()
sweep = pl.read_csv(ROOT / "leaderboard" / "official_gradability.csv")
verify_rows = [
    json.loads(line)
    for line in (ROOT / "leaderboard" / "official_verify_docker.jsonl").read_text().splitlines()
    if line
]
display(
    Markdown(f"{sweep.height} public tasks swept; {len(verify_rows)} re-graded via verify_task")
)

# %% [markdown]
# ## 2 · Gradability by repository (all 129 public tasks)
# `tools/official_gold_sweep.py` runs swegemma's `verify_sample_task_airgapped` (base run
# with the test patch, then gold patch + test patch) for every task.

# %%
by_repo = (
    sweep.group_by("repo")
    .agg(pl.len().alias("tasks"), pl.col("gradable").sum().alias("gradable"))
    .with_columns((pl.col("gradable") / pl.col("tasks")).round(3).alias("share"))
    .sort("tasks", descending=True)
)
display(by_repo)
display(Markdown(f"**Total: {int(sweep['gradable'].sum())} / {sweep.height} gradable**"))

fig, ax = plotting.figure("single", aspect=0.6)
repos = by_repo["repo"].to_list()
ax.barh(repos, by_repo["tasks"].to_list(), color="#d9d9d9", label="public tasks")
ax.barh(repos, by_repo["gradable"].to_list(), color=plotting.PALETTE[0], label="gradable offline")
ax.set(xlabel="Tasks", title="Gold patch passes in the official offline environment")
ax.legend(fontsize=6, loc="upper right")
fig.tight_layout()
plotting.save_fig(fig, "gradability_by_repo", exp_dir=REPORT, source_runs=["official_gold_sweep"])

# %% [markdown]
# ## 3 · Why tasks fail
# Failure reason per task (missing module parsed from the pytest output; "same exit" = the
# gold and base runs end with the same pytest exit code, e.g. 4 = usage error, 2 = collection
# interrupted).

# %%
display(sweep.group_by("reason").len().sort("len", descending=True))
display(
    sweep.filter(pl.col("missing_modules").is_not_null() & (pl.col("missing_modules") != ""))
    .group_by("repo", "missing_modules")
    .len()
    .sort("len", descending=True)
)

# %% [markdown]
# ### The scorer's real Phase-2 path
# `tools/official_verify_gold.py` calls `verify_task` exactly as `Evaluator.evaluate_task`
# does, with the gold patch and with an empty patch.

# %%
verify = pl.DataFrame(
    [
        {
            "instance_id": r["instance_id"],
            "gold_resolved": r["gold"]["resolved"],
            "gold_exit": r["gold"]["exit"],
            "empty_resolved": r["empty"]["resolved"],
            "gradable": r["gradable"],
            "key_line": next(
                (
                    ln.strip()
                    for ln in r["gold"]["tail"].splitlines()
                    if "Error" in ln or " passed" in ln or " failed" in ln
                ),
                "",
            )[:110],
        }
        for r in verify_rows
    ]
)
display(verify)

# %% [markdown]
# ## 4 · Reproduce (live)
# Re-grades `REPRO_IDS` through `verify_task` with the gold and the empty patch. Needs the
# official wheelhouse installed (`infra/official/.venv` here; on Kaggle, the wheelhouse
# dataset) and, for `SANDBOX = "docker"`, the `swebench-sandbox:latest` image.

# %%
if RUN_REPRO:
    out = ROOT / "data" / "probes" / f"official_verify_{SANDBOX}.jsonl"
    before = out.read_text().count("\n") if out.exists() else 0
    proc = subprocess.run(
        [
            str(ROOT / "infra" / "official" / ".venv" / "bin" / "python"),
            str(ROOT / "tools" / "official_verify_gold.py"),
            "--ids",
            REPRO_IDS,
            "--sandbox",
            SANDBOX,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print("\n".join(ln for ln in proc.stdout.splitlines() if "gold" in ln))
    new = [json.loads(ln) for ln in out.read_text().splitlines()[before:] if ln]
    display(
        pl.DataFrame(
            [
                {
                    "instance_id": r["instance_id"],
                    "sandbox": r["sandbox"],
                    "gold": r["gold"]["resolved"],
                    "empty": r["empty"]["resolved"],
                    "gradable": r["gradable"],
                }
                for r in new
            ]
        )
    )

# %% [markdown]
# ## 5 · Environment
# Versions of the official packages used for every number above.

# %%
print(
    subprocess.run(
        [
            str(ROOT / "infra" / "official" / ".venv" / "bin" / "python"),
            "-c",
            "from importlib.metadata import version as v\n"
            "for p in ('swegemma','adk-submission','adk-eval-core','google-adk','docker'):\n"
            "    print(p, v(p))",
        ],
        capture_output=True,
        text=True,
    ).stdout
)
print(
    subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "swebench-sandbox:latest",
            "--format",
            "{{.Id}} {{.Created}}",
        ],
        capture_output=True,
        text=True,
    ).stdout
)
wheels = sorted(p.name for p in (ROOT / "data" / "competition" / "wheels").glob("*.whl"))
print(len(wheels), "competition wheels; dirty_equals present:", any("dirty" in w for w in wheels))

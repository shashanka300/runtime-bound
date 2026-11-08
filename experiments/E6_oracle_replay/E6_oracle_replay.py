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
# # E6_oracle_replay — Offline context-eviction oracle
#
# **Question.** How far is the online context policy from a prefix-closed eviction oracle with a value term, computed on logged trajectories?
#
# **Exit criterion.** Oracle gap reported with CIs.
#
# Sections: 0 parameters · 1 setup & provenance · 2 preflight · 3 run · 4 live hardware
# view · 5 ingest · 6 analysis & stats · 7 paper figures · 8 teardown.

# %% [markdown]
# ## 0 · Parameters

# %% tags=["parameters"]
CONFIG_OVERRIDES: dict = {}

# %% [markdown]
# ## 1 · Setup & provenance

# %%
from pathlib import Path

from IPython.display import Markdown, display
from src.params import Params

from rbcore import load_config, plotting

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
display(
    Markdown(f"`{cfg.experiment}` · config hash `{cfg.hash()}` · sandbox `{cfg.sandbox.profile}`")
)
cfg.params

# %% [markdown]
# ## 2 · Preflight

# %%
import shutil

import httpx

from rbcore.paths import data_root

checks = {
    "data_root": str(data_root()),
    "free_disk_gib": round(shutil.disk_usage(data_root().parent).free / 2**30, 1),
}
try:
    httpx.get(f"{cfg.serving.base_url}/health", timeout=2).raise_for_status()
    checks["vllm"] = "up"
except httpx.HTTPError as exc:
    checks["vllm"] = f"down: {exc!r}"
checks

# %% [markdown]
# ## 3 · Run
# Not implemented yet — see README. Open the run with `run = RunContext(cfg, EXP_DIR).start()`,
# wrap each unit of work in `with run.phase(...)`, close it in section 8 with `run.finish()`.

# %% [markdown]
# ## 4 · Live hardware view

# %% [markdown]
# ## 5 · Ingest

# %% [markdown]
# ## 6 · Analysis & statistics
# Paired comparisons on the same tasks; `rbcore.stats.paired_diff`, `mcnemar_exact`,
# stratified by repo.

# %% [markdown]
# ## 7 · Paper figures
# `plotting.save_fig(fig, name, exp_dir=EXP_DIR, run=run)`

# %% [markdown]
# ## 8 · Teardown

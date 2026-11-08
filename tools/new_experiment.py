"""Scaffold an experiment folder with the standard layout and notebook sections.

    uv run python tools/new_experiment.py E8_my_idea --title "My idea" --question "..."

Creates README.md, config.yaml, src/params.py and a paired notebook (.py percent +
.ipynb). Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """\
# {name} — {title}

**Question.** {question}

**Hypotheses fed:** {feeds}
**Design:** {design}
**Budget:** {budget}
**Exit criterion:** {exit}

## Inputs
{inputs}

## How to run
Interactive: open `{name}.ipynb` in JupyterLab (kernel *challeenge*), run section by section.
Headless / overnight:
```bash
uv run papermill experiments/{name}/{name}.ipynb /dev/null --cwd experiments/{name} \\
  -p CONFIG_OVERRIDES '{{"seed": 1}}'
```
Every run writes `runs/<run_id>/` (manifest, logs, spans, hardware telemetry); figures
land in `figures/` with a provenance `.json` next to each file.

## Status
Scaffolded. Sections 0–2 run; later sections are implemented when this experiment starts.
"""

CONFIG = """\
experiment: {name}
description: >-
  {title}
seed: 0

serving:
  base_url: http://127.0.0.1:8000
  model: gemma-4-31b-it
  max_model_len: 32768

sandbox:
  profile: dev          # use `eval` (4 GiB / 2 vCPU) for headline numbers

params: {{}}
"""

PARAMS = """\
from pydantic import BaseModel, ConfigDict


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
"""

NOTEBOOK = """\
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: challeenge (py3.12)
#     language: python
#     name: challeenge
# ---

# %% [markdown]
# # {name} — {title}
#
# **Question.** {question}
#
# **Exit criterion.** {exit}
#
# Sections: 0 parameters · 1 setup & provenance · 2 preflight · 3 run · 4 live hardware
# view · 5 ingest · 6 analysis & stats · 7 paper figures · 8 teardown.

# %% [markdown]
# ## 0 · Parameters

# %% tags=["parameters"]
CONFIG_OVERRIDES: dict = {{}}

# %% [markdown]
# ## 1 · Setup & provenance

# %%
from pathlib import Path

from IPython.display import Markdown, display

from rbcore import RunContext, load_config, plotting
from src.params import Params

EXP_DIR = Path.cwd()
cfg = load_config(EXP_DIR / "config.yaml", Params, CONFIG_OVERRIDES)
P = cfg.params
plotting.use_style()
display(Markdown(f"`{{cfg.experiment}}` · config hash `{{cfg.hash()}}` · sandbox `{{cfg.sandbox.profile}}`"))
cfg.params

# %% [markdown]
# ## 2 · Preflight

# %%
import shutil

import httpx

from rbcore.paths import data_root

checks = {{
    "data_root": str(data_root()),
    "free_disk_gib": round(shutil.disk_usage(data_root().parent).free / 2**30, 1),
}}
try:
    httpx.get(f"{{cfg.serving.base_url}}/health", timeout=2).raise_for_status()
    checks["vllm"] = "up"
except httpx.HTTPError as exc:
    checks["vllm"] = f"down: {{exc!r}}"
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
"""


def scaffold(name: str, **fields: str) -> Path:
    d = ROOT / "experiments" / name
    (d / "src").mkdir(parents=True, exist_ok=True)
    defaults = {
        "feeds": "—",
        "design": "—",
        "budget": "—",
        "exit": "—",
        "inputs": "Competition data under `$DATA_ROOT/competition`.",
    }
    fields = {**defaults, "name": name, **fields}
    files = {
        d / "README.md": README.format(**fields),
        d / "config.yaml": CONFIG.format(**fields),
        d / "src" / "__init__.py": "",
        d / "src" / "params.py": PARAMS,
        d / f"{name}.py": NOTEBOOK.format(**fields),
    }
    for path, text in files.items():
        if not path.exists():
            path.write_text(text)
    for sub in ("figures", "tables"):
        (d / sub).mkdir(exist_ok=True)
        (d / sub / ".gitkeep").touch()
    sync_notebook(d / f"{name}.py")
    return d


def sync_notebook(py: Path) -> None:
    """Create/refresh the paired .ipynb from the percent script."""
    subprocess.run(
        ["jupytext", "--set-formats", "ipynb,py:percent", str(py)], check=True, capture_output=True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--title", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--feeds", default="—")
    ap.add_argument("--design", default="—")
    ap.add_argument("--budget", default="—")
    ap.add_argument("--exit", default="—")
    ap.add_argument("--inputs", default="Competition data under `$DATA_ROOT/competition`.")
    args = vars(ap.parse_args())
    print(scaffold(args.pop("name"), **args))


if __name__ == "__main__":
    main()

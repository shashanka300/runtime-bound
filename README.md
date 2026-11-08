# Runtime-Bound

*What limits a local 31B coding agent, and what local evaluation gets wrong.*
Code, configs and run records for our Gemma 4 Developer Agent Paper Track submission.
Everything ran on one consumer GPU (RTX 5090, 32 GB) plus one rented H100 for LoRA training.

- Paper: [`paper/`](paper/) (LaTeX source, figures, references).
- Pre-registered plan: [`docs/research_plan.md`](docs/research_plan.md).
- Every deviation from it, with dates and the evidence at the time:
  [`docs/design_changes.md`](docs/design_changes.md).
- Every leaderboard submission, the local evidence behind it and its score:
  [`leaderboard/README.md`](leaderboard/README.md).

## Where each result comes from

| Paper section | Result | Code | Records |
|---|---|---|---|
| 3 Fidelity ladder (Fig. 1) | scorer defects and local artefacts, one condition at a time | `experiments/E1b_context_fixes`, `tools/fidelity_ladder*.sh`, `tools/record_proxy.py`, `tools/replay_requests.py`, `tools/lora_canary*.{py,sh}` | `experiments/E1b_context_fixes/tables/`, `leaderboard/probes/` |
| 4.1 Accounting | 90% of starter failures are context exhaustion; output-reserve fix | `experiments/E1_starter_baseline`, `experiments/E1b_context_fixes`, `experiments/E2_token_accounting` | each experiment's `tables/` and `executed/` notebooks |
| 4.2 Phase agent | negative result under the real budget | `experiments/E5_context_architecture` | `experiments/E5_context_architecture/tables/` |
| 4.3 Wall-clock anatomy (Fig. 2) | timeouts lost in a few very long turns | `paper/make_figures.py`, `tools/official_screen.py`, `tools/overnight_screen.sh` | `paper/figures/fig2_data.json` |
| 4.4 Interventions | lb05-lb15 bundles, self-generated LoRA, weight edits | `leaderboard/build.py`, `tools/build_sft.py`, `tools/train_lora.py`, `tools/qlora.py`, `tools/modal_train.py`, `trackw/` | `leaderboard/README.md`, `docs/trackw_session1/` |
| 4.5 Measurement (Fig. 3) | public-set gradability, local vs board | `tools/official_gold_sweep.py`, `tools/official_gold_report.py`, `reports/gradability_audit/` | `leaderboard/official_gradability.csv` |

Run outputs (agent traces, telemetry, training data, adapters) are large and stay out of git
(`data/`, `experiments/*/runs/`). The tables and executed notebooks in each experiment folder are
the committed summaries. No competition data is included (see [`NOTICE`](NOTICE)).

## Layout
```
core/rbcore/        shared library: config, RunContext, telemetry, tracing, stats, plots
harness/swebox/     local SWE harness with an "official" mode matching the scorer
experiments/EX_*/   one self-contained experiment each (README, config.yaml, notebook, src/)
leaderboard/        submission bundles (built by build.py), official compile checker, probes
tools/              screens, fidelity ladder, replay proxy, LoRA data + training, analysis
trackw/             activation-guided weight-edit study (12B)
infra/              WSL setup, vLLM servers (incl. the scorer's 0.19.1 build), data download
paper/              the paper, its figures (make_figures.py) and references
docs/               research plan, design-change log, archived planning notes
data/               (gitignored) competition data, HF cache, logs, run outputs
```
**Independence rule:** experiments import `rbcore` only, never each other. Replay
experiments (E2, E6) take an `input_run` path and can generate their own input.

**Scorer fidelity.** `infra/vllm019/` installs the scorer's vLLM 0.19.1 and its pinned
dependencies from the organizers' wheelhouse with the scorer's flags; `tools/official_screen.py`
runs the official harness (swegemma) against it with the per-task cap scaled by 0.625 for the
slower 4×L4 box. It gives each wheel pool its own cache directory, because the harness caches
unpacked wheels under a name that ignores which pool built them.

## Setup (once)
1. **Windows:** copy `infra/wsl/wslconfig.example` to `C:\Users\<you>\.wslconfig`, then run
   `wsl --shutdown` (raises the WSL RAM limit from 30 GB to 56 GB). Recommended: add
   `[interop]\nappendWindowsPath=false` to `/etc/wsl.conf` so Windows Python/uv never
   shadow the Linux ones.
2. **WSL (sudo):** `bash infra/setup_wsl.sh` installs Docker Engine and the NVIDIA
   container toolkit. Open a new shell afterwards.
3. **WSL (user):** `bash infra/bootstrap_user.sh` installs uv, the research env, the
   isolated vLLM env, the Jupyter kernel and pre-commit hooks.
4. **Data:** accept the competition rules on Kaggle, put `~/.kaggle/kaggle.json` in place,
   then `bash infra/data/download.sh --models 31b,12b`.

## Daily use
```bash
bash infra/vllm/serve.sh                 # model server (own terminal)
bash infra/services.sh start             # MLflow :5000, Phoenix :6006 (optional)
uv run jupyter lab                       # kernel: "challeenge (py3.12)"
uv run python tools/run_queue.py q.yaml  # overnight headless runs
bash tools/check.sh                      # ruff + mypy --strict + pytest
```

## What every run records (`experiments/EX_*/runs/<run_id>/`)
| File | Content |
|---|---|
| `manifest.json` | git sha/dirty, config hash, package versions, lockfile hashes, GPU/driver/CPU/RAM, Docker, vLLM version + served model, data fingerprint, sandbox profile |
| `config.yaml` | fully resolved config |
| `logs.jsonl` | structured logs, every line tagged with `run_id` |
| `events.jsonl` | phase start/end, gates, user events (wall-clock ns) |
| `spans.jsonl` | OpenTelemetry spans (also sent to Phoenix when it is up) |
| `metrics.jsonl` | logged metrics (also sent to MLflow) |
| `hw/gpu` | NVML @1 Hz: util, memory, power, **energy counter**, temp, clocks, throttle reasons |
| `hw/host` | CPU, load, RAM, swap, disk IO |
| `hw/containers` | per-sandbox CPU, memory, peak memory, OOM kills (cgroup v2) |
| `hw/vllm` | vLLM Prometheus: KV usage, queue depth, prefix-cache hits, token counters, latency sums |
| `emissions.csv` | CodeCarbon energy/CO2 estimate |
| `summary.json` | status, wall time, monitor overhead, total GPU energy |

`rbcore.windows` attributes any telemetry to phases, tasks or cells (energy per task,
peak KV per cell, and so on). Figures are saved as PDF + PNG with a provenance `.json`.

## Collaborating
- `main` is always runnable: `bash tools/check.sh` must pass. Work on branches
  (`e1-harness`, `e5-phase-agent`, ...) and merge through PRs.
- Commit notebooks through the pre-commit hooks: `nbstripout` removes outputs and
  `jupytext` keeps the `.py` twin in sync, so diffs stay reviewable. Review the `.py` file.
- Never commit `data/`, `runs/` or credentials (already in `.gitignore`). Share results
  through `figures/` + `tables/` (with their provenance `.json`), or by uploading run
  folders elsewhere.
- One experiment per folder. Shared code goes in `rbcore` with a test, never
  copied between experiments.

## Sandbox profiles
`sandbox.profile: eval` reproduces the official 4 GiB / 2 vCPU containers and is
required for headline numbers. `dev` (8 GiB / 4 vCPU, the default) and `custom` give
containers more room while iterating. Every manifest records `matches_eval`.

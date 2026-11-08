# E6_oracle_replay — Offline context-eviction oracle

**Question.** How far is the online context policy from a prefix-closed eviction oracle with a value term, computed on logged trajectories?

**Hypotheses fed:** H5
**Design:** offline oracle over logged trajectories
**Budget:** ~10 GPU-h
**Exit criterion:** Oracle gap reported with CIs.

## Inputs
`params.input_run`: path to a finished run directory to replay (e.g. an E1 or E5 run).
If empty, the notebook produces its own 10-task input run first, so this experiment
never depends on another experiment having been run.

## How to run
Interactive: open `E6_oracle_replay.ipynb` in JupyterLab (kernel *challeenge*), run section by section.
Headless / overnight:
```bash
uv run papermill experiments/E6_oracle_replay/E6_oracle_replay.ipynb /dev/null --cwd experiments/E6_oracle_replay \
  -p CONFIG_OVERRIDES '{"seed": 1}'
```
Every run writes `runs/<run_id>/` (manifest, logs, spans, hardware telemetry); figures
land in `figures/` with a provenance `.json` next to each file.

## Status
Scaffolded. Sections 0–2 run; later sections are implemented when this experiment starts.

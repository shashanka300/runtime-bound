# E4_localization_tooling — Localization tooling: graph tools vs in-sandbox skill scripts

**Question.** Do in-sandbox BM25 + AST skill scripts beat the provided code-graph tools for localization with small models?

**Hypotheses fed:** H6
**Design:** {graph tools, skill scripts, both, none}; dev-40 x 2
**Budget:** ~60 GPU-h
**Exit criterion:** Localization default chosen.

## Inputs
Competition data under `$DATA_ROOT/competition`.

## How to run
Interactive: open `E4_localization_tooling.ipynb` in JupyterLab (kernel *challeenge*), run section by section.
Headless / overnight:
```bash
uv run papermill experiments/E4_localization_tooling/E4_localization_tooling.ipynb /dev/null --cwd experiments/E4_localization_tooling \
  -p CONFIG_OVERRIDES '{"seed": 1}'
```
Every run writes `runs/<run_id>/` (manifest, logs, spans, hardware telemetry); figures
land in `figures/` with a provenance `.json` next to each file.

## Status
Scaffolded. Sections 0–2 run; later sections are implemented when this experiment starts.

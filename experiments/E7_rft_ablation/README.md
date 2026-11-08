# E7_rft_ablation — Self-bootstrap rejection-sampling fine-tuning (optional)

**Question.** Does one round of QLoRA RFT on Gemma's own successful trajectories transfer to held-out repos?

**Hypotheses fed:** H7
**Design:** base vs 1 round QLoRA RFT; leave-one-repo-out
**Budget:** ~180 GPU-h
**Exit criterion:** Gain on held-out repos, not only fastapi/rich.

## Inputs
Competition data under `$DATA_ROOT/competition`.

## How to run
Interactive: open `E7_rft_ablation.ipynb` in JupyterLab (kernel *challeenge*), run section by section.
Headless / overnight:
```bash
uv run papermill experiments/E7_rft_ablation/E7_rft_ablation.ipynb /dev/null --cwd experiments/E7_rft_ablation \
  -p CONFIG_OVERRIDES '{"seed": 1}'
```
Every run writes `runs/<run_id>/` (manifest, logs, spans, hardware telemetry); figures
land in `figures/` with a provenance `.json` next to each file.

## Status
Scaffolded. Sections 0–2 run; later sections are implemented when this experiment starts.

# E3 — How should 31B spend its runtime?

**Design change (2026-09-26).** The plan's H1 compared 31B-int4 with 12B. The project goal is to
improve the 31B agent, and E2/H3 showed its failures are context exhaustion (93%), not model
capacity, so a smaller model would hit the same wall. 12B was dropped (docs/design_changes.md).

**Questions (pre-registered):** **Q-A** do fresh-context restarts help (best k>1 vs best k=1)?
**Q-B** does a larger thinking budget help (8k vs 2k, k=1)? Plus the accuracy/time Pareto front
and a recommended configuration within 360 s/task (12 h / 120 tasks). A difference counts only
if its paired 95% CI excludes 0.
**Design:** 31B-int4 × thinking {2k, 4k, 8k} × attempts {1, 2, 4} on `dev40` (fastapi 20,
rich 15, requests 5): 3 × 40 × 4 = 480 sessions, ~10 h.

## Method
- **Model.** `gemma-4-31b-it-qat-w4a16-ct` (int4). `swebox.serving.ensure_model` (re)starts an
  adapter-free vLLM server, so the sweep runs unattended; it also supports multi-model sweeps.
- **Agent held constant** (`src/variants.py`): starter prompts and tools; adapters removed
  (rank-4 no-op LoRAs; removing them frees KV-cache memory); `max_output_tokens =
  thinking + 4096`, because the starter's 16384 reserve halves the usable prompt (E2) and would
  hide the model/thinking effect.
- **Attempts** (`src/frontier.py`): 4 independent fresh-context sessions per (condition,
  task); k attempts = the first k. **deploy**: stop at the first attempt that submits a
  non-empty patch (implementable at eval time); **oracle**: pass@k (upper bound).
  Wall-clock = agent seconds actually spent.
- **Comparisons:** paired bootstrap of the resolved difference on the same tasks (stratified
  by repo) plus exact McNemar (`frontier.paired`).
- Free by-product: the t4096/k1 condition vs E1 (same thinking, starter's 16k reserve) shows
  the effect of the output-reserve fix on dev40 tasks.
- Tests: `tests/test_frontier.py` (scoring, Pareto, paired comparison, variant rewriting).

## How to run
```bash
bash tools/run_notebook.sh E3_frontier_sweep full '{}'
bash tools/run_notebook.sh E3_frontier_sweep smoke '{"params.subset": "fidelity10", "params.attempts": 2, "params.ks": [1, 2], "params.thinking_budgets": [2048, 8192]}'
```

## Caveats
Wall-clock is measured on the 5090 with 2 concurrent sessions; the eval box (4×L4, TP=4) is
slower in absolute terms, so the x-axis ranks configurations but is not the eval-box scale.

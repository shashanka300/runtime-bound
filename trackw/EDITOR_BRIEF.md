# Track W — weight-edit session brief (for the editing agent)

You are editing the weights of **Gemma 4 12B** (`gemma-4-12b-it-qat-w4a16-ct`) by hand, with no
gradient descent and no training, to remove one budget-wasting behaviour of a coding agent built
on it. Protocol after Takehi et al. ("weight-space programming", MIT CSAIL) and the Gemma 4 12B
replication by ValerioDolci/frontier-weight-edit.

## The behaviour: redundant re-fetching (`reread+dup+regrep`)
The agent fetches information it already has in its context:
- `reread`: `read_file` on a line range that overlaps a range of the same file it already read
  (and has not edited since);
- `dup`: a tool call identical (tool + arguments) to an earlier one, with no edit in between;
- `regrep`: a `grep`/`rg`/`find` with a pattern it already searched.
In the 12B baseline (39 tasks, 961 decisions) this is 30% of all calls in failed tasks and occurs
in 31 of 32 failed tasks; 14 failures ran out of the 28-call budget. Solved tasks do it too (5.7
per task), so the aim is to free calls, not to "detect failure".

The agent runs under a hard budget (28 tool calls, ~4.4 minutes per task on our GPU). Every wasted
call is a step it cannot spend on the fix. Recorded decision points (moments right before the
agent chose an action) are labelled with the action taken and its flags; `state` holds what it had
already read/run/searched at that moment.

## What you have (CPU, free)
Python in WSL: `wsl -d Ubuntu-24.04 -- <repo>/infra/vllm019/.venv/bin/python <script>`
(write scripts into your session folder: `<session dir>` in Windows is
`/mnt/f/weight_edit/session1` in WSL). In scripts:

    import sys; sys.path.insert(0, "<repo>/trackw")
    from editor_api import points, acts, weight, save_edit, evaluate, budget_left

- `points()` — dev-split decision points: `tool`, `args`, `flags`, `state`, `task`, `i`, `k`.
- `acts()` — `(ids, layers, H)`; `H[n, L, 0]` = residual stream after layer L, `H[n, L, 1]` = MLP
  input of layer L (after pre_feedforward_layernorm), at the last prompt token of point n, i.e. the
  moment before the model starts its answer (48 layers x 3840, bf16, ~700 MB; load once per
  script). `ids[n]` matches `points()[...]["i"]`.
- `weight(L, "mlp.down_proj" | "mlp.gate_proj" | "mlp.up_proj" | "self_attn.q_proj" | ...)` — the
  bf16 matrix the scorer computes with (dequantised int4), [out, in].
- MLP of layer L: `down_proj( gelu_tanh(gate_proj(x)) * up_proj(x) )`, x = MLP input; hidden
  3840, intermediate 15360, 48 layers. Note the model answers after a thinking phase, so the
  action is chosen many tokens after the position you have activations for.

## Budget: 50 evaluations (the only way to run the model)
`save_edit({"layers.L.mlp.down_proj": delta, ...}, "name")` stores an additive edit (delta has the
weight's [out, in] shape) and its rank-8 LoRA. `evaluate("name")` mounts it on the scorer's engine
(vLLM 0.19.1) and re-runs 60 target + 60 control dev decision points (temperature 0.2, seed 0;
about 3.5 minutes). It returns:
- `target_rate`: share of target points where the model again fetches redundantly
  (base model: **0.733** dev; held-out test 0.817);
- `control_flagged` (base **0.167**) and `control_tool_changed` (base **0.233**): collateral
  change on points where the recorded action was not redundant — these base values are sampling
  noise, so only rises above them are damage;
- `no_tool_call` (base 0.008): broken outputs.
With n = 60, differences under about 0.10 are noise. Each call costs 1; `budget_left()`.

## Constraints
- Do not run the model in any other way (no transformers / vLLM forwards on CPU or GPU, no
  requests to the server); `evaluate` is the only forward pass.
- Edits are additive deltas on linear weights, **rank <= 8 per matrix** (they ship as a rank-8
  LoRA; the rest is lost — `save_edit` reports the energy kept). Prefer few layers and matrices;
  the replication used 4-8 MLP neurons (about 92K parameters). An adapter also costs ~23% decode
  speed, so a big edit must buy a lot.
- No training loops, no optimiser, no gradient steps. Closed-form statistics (means, probes, least
  squares, SVD) on the provided activations are fine; reasoning about weights is the point.
- The test split is held out; you never see it. Do not modify files outside your session folder.

## Deliverable
`<session dir>\FINAL.md`: the edit name you submit, what it does mechanistically, the
dev numbers, and what you tried that failed. Stop when you are satisfied or the budget is spent.

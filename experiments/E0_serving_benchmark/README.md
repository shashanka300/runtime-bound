# E0 — Serving benchmark (gate G1)

**Question.** Can one RTX 5090 serve `gemma-4-31B-it-qat-w4a16-ct` at 32k context
correctly, and what are throughput, latency, KV headroom and energy per token?

**Feeds:** all experiments (wall-clock scaling), C1, C2 (prefix-cache effect).
**Design:** shared-prefix {0, 4096} × context {8k, 16k, 32k} × concurrency {1, 4, 8};
random tokens, fixed 512-token outputs, `--ignore-eos`, via upstream `vllm bench serve`.
**Budget:** ~8 GPU-h (in practice < 1 h).
**Exit criterion (G1):** smoke test correct, every 32k cell completes, single-stream
decode ≥ `g1_min_output_tok_s`.

## Measured per cell
| From | Metrics |
|---|---|
| `vllm bench serve` | request/output/total throughput; TTFT, TPOT, ITL, E2E (mean, p50/90/99) |
| NVML (1 Hz) | GPU energy (counter delta, J), mean power, max temperature, SM clock |
| vLLM `/metrics` (1 Hz) | peak KV-cache usage, prefix-cache hit rate |
| vLLM startup log | KV-cache capacity (tokens), max concurrency at 32k |

## How to run
```bash
bash infra/vllm/serve.sh                  # terminal 1: model server (keep running)
bash infra/services.sh start              # optional: MLflow :5000, Phoenix :6006
uv run jupyter lab                        # open E0_serving_benchmark.ipynb
# or headless:
uv run papermill experiments/E0_serving_benchmark/E0_serving_benchmark.ipynb /dev/null \
  --cwd experiments/E0_serving_benchmark
```

## Outputs
- `runs/<run_id>/results.parquet`: one row per cell, benchmark ⨝ telemetry
- `tables/e0_serving.csv`
- `figures/e0_serving_frontier.pdf`: throughput, TTFT, J/token, shared-prefix effect
- `figures/e0_timeline.pdf`: power and KV usage over the run with cells shaded

## Results (run `20260925T052414Z-39071e56-b5f1`, 94 min, 0.79 kWh GPU)
**G1: PASS.** Smoke test correct, all 32k cells complete, single-stream decode 20.3 tok/s at 32k.

| | 8k | 16k | 32k |
|---|---|---|---|
| Output tok/s, n=1 → 8 | 44 → 69 | 32 → 33 | 20 → 19 |
| Median TTFT, n=1 → 8 | 2.8 → 47 s | 6.4 → 114 s | 15 → 195 s |
| GPU J / output token (n=4) | ~8 | ~16 | ~26 |

- **KV capacity is the binding constraint.** 34,949 tokens (1.07 × 32k) with
  `gpu-memory-utilization 0.90`, `max-num-seqs 16` and the PyTorch sampler. Above 8k,
  concurrency only adds queueing.
- **Prefill costs ~2.2–2.9k tok/s.** A full 32k re-prefill costs ~15 s of wall-clock.
- **A shared prefix helps only when KV has headroom.** At 8k × 8, TTFT drops 47 → 22 s.
  At 16k/32k it has no effect, because the prefix is evicted between requests.
- Decode inter-token latency is ~18–20 ms at every setting.

Serving settings forced by one 32 GB card (all recorded in the manifest `vllm_server_env`):
text-only (`language-model-only`), LoRA off, `max-num-seqs 16`, `VLLM_USE_FLASHINFER_SAMPLER=0`
(FlashInfer's sampler needs `nvcc`).

## Caveat
The eval box is 4×L4 with TP=4. These numbers are for the local simulator only. The
5090→L4 wall-clock ratio is estimated separately by comparing a leaderboard submission
against the same configuration run locally (E1).

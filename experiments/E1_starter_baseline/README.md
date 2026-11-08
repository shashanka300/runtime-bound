# E1 — Starter baseline under realistic budgets

**Question.** How does the organizers' starter agent behave on the public tasks under 6 and
10 minute per-task budgets, and does our local harness grade correctly?

**Feeds:** H2 (where tokens go), H3 (failure modes). Every task writes a step-level trajectory
that E2 replays. **Design:** per-task time 6 min; fidelity10 → 129 tasks × 3 seeds.
**Design change (2026-09-25):** the planned 10-min condition was dropped after 227 six-minute
runs: 224 (98.7%) ended before the limit (context rejection 74%, submission 25%), so a longer
budget could change at most 1.3% of runs; the pilot showed identical 6 vs 10 min rates. The
10-min time was reallocated to later experiments.
**Budget:** ~60 GPU-h for the full run.
**Gates:** (a) harness self-check: gold patches resolve, empty patches fail; (b) G1b: the agent
runs end to end on `fidelity10`.

## How to run
```bash
bash infra/vllm/serve_submission.sh        # base model + the starter's LoRA adapters
uv run jupyter lab                         # E1_starter_baseline.ipynb, section by section
# headless, e.g. the full run with 3 seeds:
uv run papermill experiments/E1_starter_baseline/E1_starter_baseline.ipynb /dev/null \
  --cwd experiments/E1_starter_baseline \
  -p CONFIG_OVERRIDES '{"params.subset": "all", "params.seeds": [0, 1, 2]}'
```
Section 5 resumes: finished (task, seed, budget) results are skipped when re-run.

## Harness: `swebox` (harness/src/swebox)
We wrote our own implementation of the organizers' `swegemma` harness from its published
spec (`data/competition/HARNESS_README.md`). The only public re-implementation has no licence.
It has the same two-container lifecycle, the nine tools with the same JSON envelopes and limits,
budget gating and warnings, three nudge messages, patch extraction, four-pass patch apply,
test-file reset, pytest command, and ADK compaction/caching config. On top of that, every
model call, tool call and compaction is recorded (`traces/*.jsonl`), and each task is a
telemetry phase (per-task energy, KV usage and prefix-cache hits).

### Fidelity notes (what may differ from the official harness)
| # | Area | Official | swebox | Why |
|---|---|---|---|---|
| A1 | Thinking | `adk-submission` translates `thinking_config` (mechanism not published) | `chat_template_kwargs.enable_thinking` + vLLM `thinking_token_budget` | stock ADK `LiteLlm` drops `thinking_config` |
| A2 | Test dependencies | streams pre-built site-packages (contents unpublished) | resolver `assets/install_deps.py`, run at image bootstrap only: (1) deps with the repo's own specifiers from PEP 621, Poetry (non-optional), setup.cfg, setup.py, requirements files; (2) imported packages; (3) declared `pytest-*` plugins; (4) **era pinning**: newest allowed version released before the task's `created_at`, within the newest major; from PyPI if no cached wheel is that old (binary only); (5) repair coupled pins (`pip check` conflicts); (6) fill still-missing deps from PyPI under a constraints freeze | the public wheel cache is incomplete and pins are often unbounded; each rule fixed specific gold-patch failures (see below) |
| A3 | Bootstrap | per container | once per task, cached as image `swebox-task:<id>-b<N>` | same workspace, ~1 s container start |
| A4 | Prompt | "Standard Instructions 0–5" text unpublished | paraphrased from the spec's description | — |
| A5 | Patch apply | 4 passes incl. symlink-normalised prefix | passes 1, 3, 4 (snapshot symlinks are removed at extraction) | — |
| A6 | Patches in `tasks.jsonl` | — | newline-terminated before `git apply` | all 258 lack the final newline ("corrupt patch") |
| A7 | Serving | 4×L4 TP=4, 8 LoRA slots, rank 128 | 1×5090, LoRA slots/rank sized to the submission, text-only, `max-num-seqs 16`, PyTorch sampler | 32 GB card (see E0) |
| A8 | Seeds | not set | `seed` in sampling config per run seed | reproducible multi-seed runs |

The self-check (section 4) checks A2/A3/A5/A6 on every task and classifies each task
(`tables/e1_task_gradability.csv`): `gradable`, `passes_without_fix`,
`infeasible:<reason>` or `env_mismatch`.

### How the self-check shaped the resolver (gold patches resolved, of 129)
| Resolver version | Gold ✓ | What changed / what it fixed |
|---|---|---|
| competition `setup.py` only | ~40% of the sample | reads only PEP 621 deps; `requests` declares them in setup.py |
| + all declaration formats, wheel cache | 110 | pydantic 1 chosen (pydantic 2's `typing-inspection` missing from the cache) |
| + PyPI fill under constraints, newline fix | see run | `python-multipart`, corrupt patches |
| + pytest plugins, Poetry, era pinning, conflict repair | see run | `pytest-httpbin`; Pygments drift in rich; pydantic/pydantic-core pairs |

**Infeasible under the official rules** (they fail the reference patch in *any* faithful
reproduction): `requests` tasks whose target tests expect a real TCP connect timeout to
`10.255.255.1` (the sandbox is `network_mode=none`, so the kernel returns
"network unreachable" at once), and recent `fastapi` tasks whose async tests need the anyio
pytest plugin that the official command disables (`-p no:anyio`). They are kept in
official-style resolution rates and flagged in "gradable-only" rates.

## Outputs
- `runs/<run_id>/artifacts/swebox/<tag>/`: the official results layout (`summary.json`,
  `task_results.jsonl`, `patches/`, `test_outputs/`, `traces/`, `logs/`)
- `runs/<run_id>/results.parquet`, `steps.parquet`, `selfcheck.parquet`
- `tables/e1_resolved_by_budget.csv`, `tables/e1_harness_selfcheck.csv`
- `figures/e1_resolved_and_tokens.pdf`

## Results: fidelity10 (run `20260925T101854Z-16f00c2f-0789`, 35 min, 0.19 kWh GPU)
Harness gate passed (gold rate on feasible tasks 10/10). The starter agent resolved **2/10 at
both budgets** (Wilson 95% CI 6–51%): `fastapi_14303` and `rich_2725`.

**Context overflow, not time, is the binding constraint.** 16 of 20 runs ended with
`litellm.ContextWindowExceededError` after a median of ~2.5 min, far inside either budget, so
6 vs 10 min made no difference. The starter never ends its ADK invocation (it chains tool
calls), and ADK compaction runs only *after* an invocation, so compaction never fired
(0 compactions in all 20 runs) while the history grew past 32k tokens. This is direct
early evidence for H3 and the context-runtime thesis (C2).

**Root cause (found in E2):** the rejected requests are not 32k prompts. vLLM rejects when
prompt + `max_tokens` > 32,768, and the starter reserves `max_output_tokens: 16384`, so its
usable prompt is only **16k tokens**. See `E2_token_accounting/README.md`.

Other signals: prompt tokens are ~98% of all tokens and 60–95% of them are prefix-cache hits;
`search_similar_code` returned 86k characters per call on average (full source of 10
functions, uncapped by the spec), so a single call can use most of the window;
20/124 `run_command` calls returned non-zero exits.

## First observation (1 task, before the full run)
`requests_7205` was resolved in 116 s: 22 model calls, 18 tool calls, 193k prompt tokens of
which 94% were prefix-cache hits, 3.1k thinking tokens. Past thinking text is re-sent in the
history on every call (10k characters by the end), which is context cost for H2.

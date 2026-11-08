# Design changes

Every deviation from the pre-registered plan (`research_plan.md`), with date, reason and the
evidence available when it was made. Reported in the paper's reproducibility section.

| Date | Experiment | Change | Reason / evidence at decision time |
|---|---|---|---|
| 2026-09-25 | E1 | 10-min per-task budget dropped; 6 min × 3 seeds only | 224 of 227 six-minute runs (98.7%) ended before the limit (context rejection 74%, submission 25%); the fidelity10 pilot gave identical 6 vs 10 min rates. |
| 2026-09-26 | E3 | 12B comparison (H1) dropped; E3 asks how 31B should spend its runtime (thinking × fresh-context attempts) | Project goal is improving the 31B agent; E2/H3 found 93.5% of failures are context exhaustion (a 16k usable prompt from `max_output_tokens: 16384`), which a smaller model would hit equally. |
| 2026-09-26 | E3 | Agent held constant with `max_output_tokens = thinking + 4096` and no LoRA adapters | The starter's 16k output reserve halves the usable prompt (E2) and would mask thinking/attempt effects; the starter's adapters are rank-4 on one layer. |
| 2026-09-26 | E1b (new), E3 | Context-fix ablation E1b inserted before E3; E3 postponed until a fixed agent exists | Review after E1/E2: the cheapest, highest-leverage lever (the output reserve) had not been measured, and E3's thinking/attempt effects depend on post-fix behaviour. E1b arms: reserve fix, prompt guidance, fresh-context retry (diagnostic). |
| 2026-09-26 | planned (week 3–4) | External transfer study with Ternary Bonsai 2 27B (PrismML; Qwen3.8 27B, ternary 1.72 b/w, llama.cpp fork), decided before any Bonsai result | Generality criterion: test whether the context-budget diagnosis and the C2 method transfer to another model family/quantization. Scheduled after the C2 agent exists so the final method, not interim fixes, is tested. Reported as within-model effects (fix vs no fix; 32k vs larger context), never as a head-to-head row: model family, base strength, quantization and engine differ. Not submittable (competition requires Gemma 4). |
| 2026-09-26 | planned (week 3–4) | Transfer study revised: **primary** = Qwen3.8 27B **int4 (W4A16/AWQ) on vLLM**; Bonsai 2 only as an add-on compared against that Qwen-int4 | Qwen-int4 changes only the model family (same precision class, similar VRAM/KV regime, same engine and harness), so "the method transfers" is attributable. The official full-precision (54 GB) and FP8 (~28 GB, no KV room) checkpoints don't fit usefully on 32 GB; candidate checkpoints: amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16, cyankiwi/Qwen3.8-27B-AWQ-INT4 (verify load + smoke accuracy first). Bonsai vs Qwen-int4 then isolates compression+engine on the same base model ("does compression that frees KV room beat an accurate but context-starved model?"). |
| 2026-09-26 | planned (week 3–4) | Transfer study split by engine: (1) Gemma 31B-int4 vs Qwen3.8 27B AWQ-int4, both on **vLLM** (isolates model family; primary); (2) Ollama `qwen3.8:27b` (18 GB GGUF, llama.cpp) vs Ternary Bonsai 2 (PrismML llama.cpp fork), both **llama.cpp** (isolates compression on the same base model; add-on) | Keeps each comparison within one engine. Ollama lacks vLLM's Prometheus metrics (no prefix-cache/KV telemetry) and silently truncates prompts beyond `num_ctx`, so `num_ctx=32768` must be set explicitly and recorded; run Ollama inside WSL (one GPU, one model at a time). |
| 2026-09-26 | planned (week 3–4) | Transfer study scope finalised: **only** Gemma 31B-int4 vs Qwen3.8 27B AWQ-int4 on vLLM. The Bonsai 2 / Ollama llama.cpp add-on is dropped | User decision: the family-transfer comparison is sufficient for the generality claim; the compression question is out of scope. |
| 2026-09-26 | E5 | Phase-agent variant for the pre-registered G3 test: **phase_v2_2**, chosen before its result | Smoke tests (fidelity10, seed 0; noisy, used only to catch gross problems): v1 2/10 (8/10 out of time; localizer 65% of time), v2 4/10 (no reproducer, localizer thinking off; one localizer repeated a grep 322 times), v2.1 2/10 (localizer thinking 1024 restored slow exploration). v2.2 = v2 + a "never repeat an identical tool call" prompt rule. All variants remain buildable (`src/build_submission.py`) and are reported as the development path. |

## 2026-09-26 — E5 G3 run: early stop at 60 of 80 task-runs (user decision)
At 47 paired task-runs, phase_v2_2 was level with arm B (14 vs 13 resolved, +2.1 pts). The user asked
to cap the run at 60 task-runs and drop it unless it is improving. Rule applied by
`tools/early_stop.sh`: at >= 60 finished task-runs, stop the run unless the paired gain vs arm B is
>= +5 pts (the pre-registered G3 threshold). A stopped run is analysed on the pairs it finished:
seed 0 complete (40 tasks), seed 1 partial. Consequence: the G3 test runs on n~60 unbalanced pairs,
not 80, and is reported as such.

## 2026-09-26 — E5 loop guard: sampling, not harness code (phase_v2_3)
G3 v2.2 trajectories: 28/79 task-runs repeat one identical tool call >= 10 times (max 335, an
empty grep), vs 1/80 for arm B; resolved 21/51 (41%) without a loop vs 6/28 with one. The user
chose a loop guard. A guard in harness code (blocking/rewriting repeated calls) is not
submittable: the spec compiles YAML only against closed tool/callback registries, and vLLM's
frequency/presence penalties only see the current completion. The submittable lever is sampling:
the starter's temperature 0.2 is near-greedy, and loops sit in the thinking-off localizer (17) and
the barely-thinking editor (11). phase_v2_3 = v2.2 + Gemma's recommended T=1.0 / top_p 0.95 /
top_k 64 in those two phases (verifier stays at 0.2). Loops are mostly per-run, not per-task
(seed-0 loopers re-looped on seed 1 in 5/17), so the test uses dev40 seed 0 (v2.2: 17/40 looped),
not the looping tasks, to avoid regression to the mean.

Outcome of the 60-task-run check above: gain was exactly +5.0 pts at 60 (21 vs 18), so the run
continued to 80 as the rule says.

## 2026-09-26 — E1b arm E: the sampling control for E5 v2.3
phase_v2_3 changed sampling (T 0.2 -> 1.0) while arm B kept the starter's 0.2, so v2.3 vs B
confounds architecture with sampling. Arm E = B + the same Gemma-recommended sampling
(T 1.0, top_p 0.95, top_k 64), run on dev40 seed 0 (the pairs v2.3 ran), with arms A and B
reused from their earlier runs (new `reuse` param) and outputs suffixed `_armE`.
Comparisons: E-B (does sampling alone help the monolithic agent?) and E-A; v2.3 vs E is read
from the paired tables afterwards.

## 2026-09-27 — Transfer-gap ladder (leaderboard #1 = 0.03 vs local 31%)
lb01 scored 0.03 (2/58). Instead of more agent work, local runs are moved toward the scorer one
condition at a time (cumulative ladder, dev40 seed 0, paired with arm B):
1. arm F = B without LoRA (the scorer's patched vLLM 0.19.1 wipes adapters; forum 743508);
2. arm F on the scorer's engine, vLLM 0.19.1 from the official wheelhouse (infra/vllm019);
3. + the scorer's thinking translation (see next entry).
Kept as E1b arms/params rather than a new experiment folder because every step is an arm of the
same starter-based agent; a dedicated experiment can collect the ladder for the paper.

## 2026-09-27 — Fidelity assumption A1 was wrong: the scorer sends no thinking budget
The official wheels (wheelhouse: adk_submission 0.2.11, swegemma 0.2.7) show that
`apply_thinking_config_to_model` sends only `chat_template_kwargs.enable_thinking` (+
`reasoning_effort` for low/medium/high); `thinking_budget` is validated but never reaches vLLM.
So on the scorer thinking is bounded only by `max_output_tokens` (16384 in the starter, 8192 in
lb01), while every local run so far (A1: `thinking_token_budget`) capped it at 4096. swebox now
has `ServingTarget.thinking_translation` ("budget" = A1, default so earlier runs reproduce;
"official" = the scorer's mapping). The host's getting-started notebook also strips
`thinking_level` and serves with `default_chat_template_kwargs={'enable_thinking': True}`.
Local results before this date overstate what the same submission achieves on the scorer.

## 2026-09-27 — Incident: arm F restarted after a self-inflicted server outage
Stopping the first ladder orchestrator (while it only waited) fired its EXIT trap, which stopped
the vLLM server under the running arm F; ~10 of 15 finished tasks ended in InternalServerError.
Arm F was stopped and rerun from scratch (the broken run 20260926T213137Z-49fe99b9-cd8e is kept
but excluded). Fix: tools/fidelity_ladder.sh arms its restore trap only after it has replaced
the server itself.

## 2026-09-27 — Ladder revised: the scorer's engine rejects a thinking budget
Step 1 (arm F, no LoRA, vLLM 0.30): 30.0% = arm B (F-B 0.0 [-10, +10]); LoRA is not the gap.
Step 2 as planned (vLLM 0.19.1 + our "budget" mapping) is impossible: 0.19.1 answers every
request carrying `thinking_token_budget` with "reasoning_config is not configured"; the scorer
does not set --reasoning-config, which independently confirms it never budgets thinking. That
run (20260926T234243Z-f77eec3d-d931, 40 instant failures) is kept but excluded. The ladder is now
a 2x2 on arm F: {vLLM 0.30, 0.19.1} x {budget, official}, minus the impossible cell:
0.30/budget (step 1), 0.19.1/official (the scorer's setup), 0.30/official (engine isolation).
Environment: the first 0.19.1 start failed because uv resolved transformers 5.17.0 (and newer
tokenizers, mistral-common, xgrammar) from PyPI; infra/vllm019 now installs every serving wheel
from the wheelhouse with --no-deps at the pinned versions (transformers 5.13.1), as the host's
getting-started notebook does.

## 2026-09-27 — Ladder result: the engine, not the thinking translation, drives the gap
Arm F (no LoRA), dev40 seed 0, 6 min, concurrency 2:
| engine / thinking | resolved | median task s | submitted | median compl tok/call | median latency/call |
|---|---|---|---|---|---|
| vLLM 0.30 / budget   | 30.0% | 142 | 45% | 107 | 2.7 s |
| vLLM 0.30 / official | 30.0% | 149 | 60% |  98 | 2.4 s |
| vLLM 0.19.1 / official (scorer) | 17.5% | 360 (all time) | 15% | 207 | 4.4 s |
Translation effect 0 (on 0.30 the model thinks ~40 tokens per call either way, so the 4096
budget never bound). Engine effect -12.5 pts: 0.19.1 generates ~2x tokens per call (usage
reports no reasoning tokens there), so tasks hit the time cap. The scorer's 4xL4 decodes ~1.6x
slower still (34.1 vs ~55 tok/s), which compounds toward the low single digits on the board.
Hypothesis for the mechanism (to test): 0.19.1's Gemma 4 template/parser acts on the
reasoning_effort="high" the scorer sends, or thinks longer by default. Consequence: local
evaluation of leaderboard candidates now runs on infra/vllm019; arms G (thinking off) and H
(thinking capped by max_output_tokens 4096) run there next.

## 2026-09-27 — Kaggle-fidelity mode and what it changed
"Fidelity mode" = scorer engine (vLLM 0.19.1, pinned wheelhouse), scorer thinking translation,
no LoRA, 3.1 min per task (the scorer's 5 min at the cited ~1.6x slower 4xL4 decode, 34.1 vs ~55
tok/s), tasks one at a time. Results on dev40 seed 0:
| config | resolved | median/mean task s | note |
|---|---|---|---|
| G thinking off, max_output 4096 (= lb03) | 32.5% | 78 (median) | best; submitted as lb03 |
| H thinking on, max_output 4096 | 27.5% | 186 (median) | thinking fills the short budget |
| E5 phase_v2_3 | 25.0% | 148 (mean) | vs G -7.5 [-17.5, +2.5] |
| E5 phase_v2_4 (thinking off in every phase) | 17.5% | 119 (mean) | vs G -15 [-27.5, -2.5]: worse |
Under the scorer's conditions the phase architecture loses to the single agent: the localizer
alone makes 1880 of 2568 calls (peak prompt 28.5k) and the phases split a budget that is already
short. The paper reports this as a negative result for C2 under realistic budgets.
Engine probe (first agent turn, identical prompts, 12 dev40 tasks): vLLM 0.30 and 0.19.1 are
identical (greedy completion tokens match on most tasks; median 172.5 vs 172.5; reasoning_effort
has no effect, the Gemma 4 template ignores it). The ~2x longer calls on 0.19.1 must therefore
come from later turns (history rendering); tools/archive/queue_replay.sh records real multi-turn requests
on 0.30 and replays them greedily on both engines.
Harness change: grading runs outside the agent slots (own pool of 4) and rich_4006 (always hits
the 30 min grading cap) is scheduled first; the telemetry "task" phase no longer includes grading.

## 2026-09-27 — Mechanism: the scorer's engine drops the agent's earlier thoughts
Replay (tools/archive/queue_replay.sh): 164 requests from 4 real sessions (lb02 config, recorded through
tools/record_proxy.py on vLLM 0.30) replayed greedily on both engines. ADK/LiteLLM sends every
earlier assistant turn with its `reasoning_content`. vLLM 0.30 renders it into the prompt; vLLM
0.19.1 drops it. Same request, different prompt: identical on the first turn (no history), then
0.19.1 is shorter by a median 418 tokens at turns 2-4 and 2,583 at turn 5+ (only 5% of pairs have
equal prompt lengths). Without its own earlier reasoning the model re-derives its plan each turn:
later turns generate more (e.g. turn 10: 378 vs 55 tokens) and take longer. This explains the
whole ladder: first-turn probes identical; thinking-on arm F 30% -> 17.5% on 0.19.1; thinking-off
arm G unaffected (32.5%: nothing to drop). Submittable remedy to test: make the agent keep a
short plan in its visible message text (content is kept by both engines) instead of only in
thoughts.

## 2026-09-27 — CORRECTION: local KV-cache stalls inflated the measured engine effect
The replay hung on a single request (Waiting 1, Running 0, KV 0%). vLLM 0.19.1 on the RTX 5090
allocates only 10,496 KV tokens (0.19 enables CUDA-graph memory profiling by default) vs 45,506
for vLLM 0.30; the scorer's 4xL4 has ~88k (cited). Audit of every 0.19.1 run for tasks that end
waiting >60 s on an unanswered model request:
| run | concurrency | stalled tasks | max call latency |
|---|---|---|---|
| arm F, ladder step 2 (6 min)   | 2 | 26/40 (median wait 230 s) | 328 s |
| arms G / H (6 min)             | 2 | 3 / 16 | 296 s |
| arms G / H at L4 speed         | 1 | 0 / 0  | 25 s / 73 s |
| E5 v2.3 / v2.4 fidelity        | 1 | 0 / 0  | 75 s / 61 s |
| arm F on vLLM 0.30 (reference) | 2 | 0-1    | 171 s |
So the ladder's "engine -12.5 pts" (arm F 30% -> 17.5%) mixes a real effect (0.19.1 drops earlier
reasoning_content) with a local artifact the scorer does not have (KV stalls at concurrency 2 on
a 32 GB card). That number must not be reported as the engine effect. Unaffected: every
concurrency-1 run, i.e. the L4-speed G/H comparison, the E5 fidelity runs, and the lb03
prediction. Next: give 0.19.1 a scorer-like KV budget locally, re-measure arm F at concurrency 1
and L4 speed, and finish the replay.

## 2026-09-27 — Clean engine effect is ~0 on accuracy; the board gap is still unexplained
0.19.1's KV accounting could not be changed locally (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
leaves 10,496 tokens; the setting was reverted). Instead every engine comparison now runs at
concurrency 1, which had no stalls. The replay was stopped after 64 of 164 pairs (one long
request never scheduled); those pairs suffice for the history-rendering result.
Arm F (thinking on, max_output 8192, no LoRA, official translation), L4 speed, concurrency 1:
| engine | resolved | timeouts | context deaths | median task s | stalls |
|---|---|---|---|---|---|
| vLLM 0.19.1 | 27.5% (11/40) | 22 | 12.5% | 186 | 0 |
| vLLM 0.30   | 25.0% (10/40) | 5  | 35%   | 125 | 0 |
Paired 0.30-0.19.1: -2.5 [-15, +10], p = 1.0. The engines fail in opposite ways: 0.30 keeps
earlier thoughts (prompts grow, context deaths), 0.19.1 drops them (re-thinking, timeouts).
So, stall-free, LoRA (0), thinking translation (0) and engine (~0) do not explain lb01's 0.03:
locally lb01's configuration lands at ~25-30% under L4-speed scaling. Remaining candidates,
named in the lb03 pre-registration: L4 prefill speed (cited ~210 tok/s at concurrency 1; our
3.1 min scaling models decode only) and harder hidden tasks. lb03's score separates them.
Standing results: thinking off (G) is the best and fastest configuration on the scorer engine
(32.5%, median 78 s/task); the phase agent loses under the short budget (v2.3 25%, v2.4 17.5%).

## 2026-09-28 — lb03 = 0.06 (4/58); official-harness gradability audit (in progress)
lb03 scored 0.06, refuting the pre-registered 12-27/58. Time-budget curve of G on the scorer
engine (concurrency 1): 3.1 min 32.5%, 1.0 min 17.5% (2.0 min interrupted at shutdown). Even at
1 min the local rate is far above the board, so speed alone does not explain the gap.
Gradability under the OFFICIAL harness: tools/official_gold_sweep.py runs swegemma's
verify_sample_task_airgapped (network none, sandbox image swebench-sandbox:latest, competition
wheels) with the gold patch on each public task. 111/129 done before shutdown: 27 gradable
(rich 25/30, httpx 1/1, fastapi 1/67, requests 0/13); swebox's own selfcheck found 114/129.
FastAPI tasks fail because `starlette` cannot be imported (46 tasks), then dirty_equals (4),
anyio (1); requests fails with pytest usage errors (exit 4). starlette IS in the competition
wheels (many versions), so the failure is in how this code path installs the package ("setup.py
not found" at the editable install; FastAPI builds from pyproject.toml). Re-scoring G (L4 speed)
on the officially gradable tasks only: 2/11 solved = 5.0% of all tasks, vs lb03's 6.9% on the
board: consistent with the gap being a grading-environment cap. NOT yet established: whether the
scorer's real Phase-2 path (verify_task inside Evaluator) fails the same way; this helper may
differ. Report: leaderboard/official_gradability.csv (tools/official_gold_report.py).
Next (docs/archive/NEXT.md): gold patches through the real verify_task path; finish the sweep; finish G
at 2 min; then decide on a forum report (user's call) and on re-basing local evaluation.

## 2026-09-28 — The local-to-board gap is the grading environment
Real Phase-2 path (tools/official_verify_gold.py: swegemma verify_task exactly as Evaluator
calls it, docker sandbox, official image, competition wheels): with the GOLD patch, fastapi
tasks fail collection on `ModuleNotFoundError: dirty_equals` (and starlette imports in the
sweep), requests has 187 errors (fixture package missing), rich grades correctly
(leaderboard/official_verify_docker.jsonl).
Full sweep (tools/official_gold_sweep.py, leaderboard/official_gradability.csv): 41/129 public
tasks gradable offline: rich 39/48, httpx 1/1, fastapi 1/67, requests 0/13 (swebox, which fills
missing packages from PyPI at image build, grades 114/129).
Re-scored on officially gradable tasks only (tools/official_gold_report.py):
| config | raw local | as share of all tasks, gradable only |
|---|---|---|
| G, L4 speed 3.1 min (= lb03) | 32.5% | 7.5% (3 of 15 gradable) |
| G, 1.0 min | 17.5% | 2.5% |
lb03 on the board: 4/58 = 6.9%. The corrected prediction matches; LoRA, thinking translation,
engine and decode speed are each ~0 once stalls are removed. Caveat: the scorer may run the
subprocess sandbox in Kaggle's notebook image and the hidden set may use other wheels; we cannot
observe that, but the agreement with lb03 says the cap is present on the board.
Consequences: (1) every team is capped; the achievable board score is set mostly by rich (and
httpx) tasks; (2) local evaluation must score on officially gradable tasks only (or reproduce
the official environment) to predict the board; (3) agent work should target the gradable task
types; (4) paper headline: the transfer gap decomposition ends in the benchmark's grading
environment.
Time-budget curve of G on the scorer engine (concurrency 1), raw / gradable-only share of all
tasks: 1.0 min 17.5% / 2.5%; 2.0 min 25.0% / 5.0%; 3.1 min 32.5% / 7.5%. The L4-equivalent
point matches lb03 (6.9%), so the decode-speed scaling holds and no residual factor remains.
Evidence notebook with a draft forum post (not posted): reports/gradability_audit/.

## 2026-09-28 (afternoon) — CORRECTION: the gradability cap does not apply to the leaderboard
Forum, host (Ryan Holbrook) in thread 742882: "the evaluation metric correctly validates a
gold-patch submission at 100%", and "the private Docker sandbox uses the same base image but
installs a different set of dependencies because the test issues come from a DIFFERENT SET OF
REPOS"; the public-environment failures are likely a subprocess/public-wheels problem. Also
743800: the wheelhouse is the authoritative package set; the scoring environment is fixed.
So the hidden tasks come from other repositories, graded in an environment where every gold
patch passes. The 41/129 public gradability result stands for the PUBLIC development set only
(it makes the official public tooling unreliable as a positive control, confirmed by others:
missing dirty_equals / inline_snapshot / typing_inspection, requests SSL failures on py3.13
with pytest-httpbin's certificate, installed packages shadowing /workspace), but it does NOT
explain the board. The "7.5% vs 6.9%" agreement is a coincidence and is withdrawn as evidence.
Also found: swegemma 0.2.7 caches unpacked wheels in $TMPDIR/swegemma_sp_cache_v8 with a key that
ignores the wheel set (thread 744029). Our first 111 sweep rows ran after a no-wheels attempt had
filled that cache, so their starlette failures are an artefact; a clean sweep (cache cleared)
is running -> data/probes/official_gold_v2.jsonl.
What remains for the board gap: the hidden tasks are from repositories our agent has never seen
and were filtered to be frontier-model-feasible (thread 743817), i.e. a distribution shift plus
difficulty, which no local public-task measurement captures. Everything measured stall-free
still holds (LoRA, thinking translation, engine ~0; thinking off best; phase agent worse).
Other forum facts: provided embeddings look near-collapsed (744040: gold file in top-10 for
35/129 tasks via embeddings vs 67 via grep) -> relevant to E4.

## 2026-11-04 — LoRA canary before/after the scorer's vLLM patch
tools/lora_canary_before_after.sh: the starter main_lora is a no-op by construction (rank 4,
layer 0 q/o_proj only, every lora_B = 0). A canary with the same layout and lora_B ~ N(0,1) x 50,
greedy, one prompt: on the wheelhouse vLLM 0.19.1 of 2026-09-25 the canary output is identical
to the base model (text and top-5 logprobs: weights wiped); on the patched build of 2026-09-27
the canary changes the text ("mysteriouserious ... saltwater water") and the logprobs
(-13.125 -> -12.19). Results: leaderboard/probes/lora_canary_{old,new}_wheel_s50.json.

## 2026-11-04 (night) — vLLM YAML config drops boolean false: reserve-full-isl was never off
Overnight screen (tools/overnight_screen.sh, official harness V31 + vLLM 0.19.1) stalled on its
first task: "Running: 0 reqs, Waiting: 1 reqs, KV usage 0%". The engine reports 7,968 KV tokens
on one 32 GB card, and `scheduler-reserve-full-isl: false` in infra/vllm019/serve.yaml was
absent from vLLM's "non-default args" (default True): vLLM's --config YAML loader turns `true`
into a flag and silently drops `false`. Fix: serve.sh passes --no-scheduler-reserve-full-isl on
the command line (as the scorer's VllmServer does). Consequence for the September record: every
0.19.1 run we did had full-ISL reservation ON, i.e. the "KV stall" at concurrency 2 is this same
scheduler behaviour, which the host's V31 server now disables. Screen restarted from task 1.

## 2026-11-05 00:00 IST — screen parked for Track W; 12B serving needs the host's hf-overrides
User parked the leaderboard screen (V1 10/39 rows kept, 3 resolved; V2 0 rows) to start Track W.
The 12B checkpoint declares Gemma4UnifiedForConditionalGeneration (gemma4_unified); vLLM 0.19.1
falls back to its Transformers backend and dies in profile_run (self_attn.v_norm: "Expected
hidden_size to be 3840, but found: 256"). Host fix (discussion 748287): --hf-overrides
'{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}' (in serve_12b.yaml); the
follow-up vision_embedder KeyError is fixed in wheelhouse V31 (still the latest, 2026-10-12).
The 12B baseline run (lb06_12b) goes through tools/record_proxy.py so its requests can be replayed
as decision points on edited weights. W1 scan of V1 (31B, 10 tasks, tools/behaviour_scan.py):
redundant calls are ~12% of calls in failed tasks (overlapping re-reads 15, duplicates 9);
thinking is short (5-13k completion tokens per task); failures are budget exhaustion; the most
failure-linked behaviour is a late first edit (call 29 of 32 in rich_4075; 2 tasks never edited);
2 calls used a malformed key ("start_line\"") that the tool ignores. Also seen on the forum:
"Scoring zero when using LoRA adapters" (Luca Massaron, open) -> check before any LoRA submission.

## 2026-11-05 00:20 IST — Track W harness built and validated (CPU)
- transformers (scorer env, transformers 5.13.1) loads the 12B QAT checkpoint and matches vLLM
  greedy decoding exactly (64/64 chars; trackw/load_check.py). Linears stay int4-packed and are
  dequantised inside the full-model forward (calling a submodule alone fails), so edits are
  applied as output hooks in transformers and as LoRA in vLLM.
- trackw/weights.py dequantises any linear straight from safetensors (int4 sym, group 32); output
  matches the real forward to 0.17% (bf16 rounding).
- vLLM renders chat requests with a space before <|tool> that apply_chat_template does not, so
  decision points use vLLM's own /tokenize ids (trackw/dump_tokens.py; 39/39 = usage counts).
- Chunked prefill (4k chunks + KV cache) == single pass (cosine 1.00000), so the all-layer
  activation pass (residual + MLP input, 48 layers, bf16) is memory-bounded.
- ADK events compaction is active (summary replaces history ~every 15 events); summariser calls
  are excluded from decision points; points after compaction are marked.
- Caveat: CPU checks (12B forwards) overlapped 12B baseline tasks rich_4077 and rich_4075 (both
  hit the time cap; load average ~5.6/24 cores); no more CPU-heavy work during the baseline.

## 2026-11-05 00:45 IST — Track W1 decision: target = 12B thought-channel loop (not re-reads)
Gemma 4 12B (lb06_12b baseline, first 5 tasks, scorer engine and flags) degenerates into
"<|channel>thought\n<channel|>" repeated until the token limit: in agent steps (8,192 tokens,
70-93 s each) and in ADK events-compaction summariser calls, which have no output cap (25,315 and
20,661 tokens, 268 s and 218 s: one such call is the whole 4.4-min task budget). Also milder:
reasoning that repeats itself until the 4,096 thinking budget. So far 3 agent steps + 1 summariser
call in 2/5 tasks, 648 s of generation lost. 31B shows nothing similar (5-13k completion tokens per
task). GO for Track W with this target (re-reads were the 31B picture). First question, queued
for the GPU window: model or engine? trackw/loop_probe.py replays the looping requests with the
recorded settings, without thinking_token_budget, at temperature 1.0, and with thinking off.
Hypothesis to test: once the thinking budget is spent, vLLM forces <channel|> each time the model
reopens a thought -> open/forced-close loop (a runtime artefact, fixable by configuration).

## 2026-11-05 02:45 IST — 12B baseline 7/39; CORRECTION to the 00:45 target choice
lb06_12b (12B, lb04 recipe, official harness V31, 7 min x 0.625): 7/39 resolved (18%); 31B on
the same setup had 3/10 on its first 10 tasks (V1, parked). Full 12B decision-point data: 961
points, 39 tasks (data/screen_nov/lb06_12b_requests.points.jsonl). Applying the pre-set go/no-go
rule (behaviour in >= 25% of failed tasks) to the FULL data instead of the first 5 tasks:
  re-read of already-seen lines 31/32 failed tasks (139 calls); exact duplicate 30/32 (119);
  repeated grep 12/32 (26); thought-channel loop 6/32 = 19% (8 agent steps + 2/53 summaries,
  1,335 s of generation) -> below the bar.
Failed tasks average 7.1 redundant calls of 23.9 (30%); 14/32 failures hit the 28-call cap,
20 the time cap. Solved tasks also average 5.7 redundant calls (not discriminative; the value is
freed calls under the cap). DECISION: edit target = redundant re-fetching family
"reread+dup+regrep" (eval_points/editor_api default). The 00:45 choice (channel loop) was made
on 5 tasks and is withdrawn; the loop stays a separate finding (probe still queued: model vs
engine). Also fixed: tools/screen_12b.sh restore failed ("uv: not found", PATH in setsid shell).

## 2026-11-05 05:55 IST — Track W GPU results; editor session 1 started; PRE-REGISTERED prediction
- Activation pass done: data/screen_nov/lb06_12b_requests.acts.pt (residual + MLP input, 48
  layers, 961 points; 2.7 h on GPU: compressed-tensors re-dequantises every layer per chunk).
- Loop probe (10 recorded looping requests x 3 seeds, 4096 new tokens): loop rate 11/30 as
  recorded, 11/30 without thinking_token_budget, 11/30 at temperature 1.0, 13/30 with thinking
  OFF. The budget-enforcement hypothesis is REFUTED: the empty thought-channel loop is a 12B
  model behaviour in these contexts (a second, MIT-style edit target; kept as a separate finding).
- LoRA on vLLM 0.19.1 + 12B: adapter is applied (base vs adapter top-20 logprobs move 1.7-3.0;
  merged-delta reference closer to the adapter, 0.64-0.68, than to base, 1.1-3.1; residual gap
  from raw-completion BOS handling, not chat requests). Decode speed with an adapter mounted:
  144 -> 112 tok/s (-23%), consistent with forum reports that LoRA slows scoring runs.
- Base 12B on decision points (trackw/eval_points.py, 60 target + 60 control, temp 0.2 seed 0):
  dev target_rate 0.733, control_flagged 0.167, control_tool_changed 0.233, no_tool_call 0.008;
  test target_rate 0.817, 0.117, 0.233, 0.033. Editor API smoke test (tiny edit): 0.700 (null).
- Editor session 1: the standalone claude CLI is not logged in on this machine (login is the
  user's), so the approved session runs as an Opus 5.5 max-effort background subagent with the
  same brief (trackw/EDITOR_BRIEF.md), workspace <session dir>, 50 evaluations.
PRE-REGISTERED (before any edit result): the best edit lowers dev target_rate from 0.73 to
0.55-0.65 with control damage within noise; on the held-out test split the drop is smaller; end
to end (39 tasks, edited vs base 12B) task success is unchanged within noise (7/39 +- 3), because
redundant calls are not what separates solved from failed tasks and the adapter costs ~23% decode
speed. A drop below 0.45 on test with no control damage would falsify the "edits can't buy
budget" expectation.

## 2026-11-05 10:30 IST: Track W session 1 closed, a NULL result (no edit submitted)
- 6 of 50 evaluations used; the editor stopped early, judging the allowed routes exhausted (docs/trackw_session1/FINAL.md).
  Dev target_rate: base 0.733; e01 0.700, e02 0.700, e04 0.683, e06 0.700 (all null); e03 0.683 with 85%
  of outputs lacking a tool call (broken); e05 0.550 with 29% lacking a tool call and 57% of controls changing tool (degraded,
  not steered). Any mounted near-null adapter sits at about 0.70.
- Pre-registered prediction (best edit 0.55-0.65 on dev, no end-to-end gain): CONSISTENT. No edit cleared
  the dev bar without control damage, so the test split and the end-to-end run were not used (they are
  only spent on a candidate).
- Mechanistic findings for the paper: 32/60 dev targets re-fetch content that context compaction removed,
  so re-reading is rational and the ceiling for an in-context fix is about 0.45; the readable "commit/explore"
  and "Wait" directions at the decision token are correlates, not levers (no effect at 2.5 sd; at 10.8 sd the
  model breaks before it steers); the flagger misses sed/cat/head re-reads (an edit could game target_rate).
- Implication: the redundancy is a runtime/harness property (compaction drops edited-file content;
  budget warnings are ignored), not a weight-level habit. That supports the runtime-bound thesis.
- Session 2 not started: the editor reports a selective edit needs activations mid-thought (at the last
  plan sentence and at the tool-name token), which needs a new multi-hour activation pass. Deferred; V1 resumes.

## 2026-11-05 18:15 IST: V1 screen done; lb04 scored on Kaggle
- Local official-harness screen (31B, vLLM 0.19.1 V31 flags, 39 gradable public tasks, time x0.625):
  lb04_public_7min 16/39 (41%), lb05a_think2048 12/39 (31%), lb05b_audit_rules 14/39 (36%).
  Neither lb05 variant beats lb04 locally, so there is no new submission candidate from V1.
- Kaggle: lb04 (ref 57993052) COMPLETE, public 0.15 (lb03: 0.06; top: 0.29). Local 41% vs hidden
  0.15 again shows the public-to-hidden transfer gap (see the transfer-gap notes).

## 2026-11-05 21:10 IST: lb07_safe_tools screen: no win
- lb07 (graph tools removed, 36 calls / 6 min, temp 1.0, ranged-read + docstring-safe edit rules):
  14/39 vs lb04 16/39. Paired: both 13, lb04-only 3, lb07-only 1 (noise level, not a win).
  Mean Kaggle-equivalent time 5.37 min (lb04 5.48), so it fits 12 h. Time-capped tasks rose 15 -> 23: the
  6-minute cap binds harder than the extra calls help (call-capped 14 -> 12).
- Not submitted. The freeze protection (no array-arg tool) cannot be measured locally; the lb04 prompt
  already forbids the graph tools.

## 2026-11-06 — lb08 (notes file) and lb09 (skill helpers): both below lb04; lb10 = planning on lb04

39-task official screen, same tasks as lb04 (16/39, mean 5.48 Kaggle-min).

| variant | resolved | mean Kaggle-min | call-capped | time-capped | empty patch | paired vs lb04 |
|---|---|---|---|---|---|---|
| lb04_public_7min | 16 | 5.48 | 14 | 15 | 11 | — |
| lb08_notes | 14 | 5.88 | 18 | 17 | 12 | 13 both, 3 only lb04, 1 only lb08 |
| lb09_skill_tools | 13 | 5.59 | 13 | 18 | 17 | 11 both, 5 only lb04, 2 only lb09 |

- lb08: writing /tmp/notes.md costs calls and time (5.88 min breaks the ~5.5 min budget for 12 h) with no gain.
- lb09: helpers were used on almost every task (run_skill_script/install.py, locate/outline/replace all in
  ~37/39 traces, no NO_CODE_EXECUTOR errors), so the skill path works mechanically. But empty patches rose
  11 -> 17: more time goes into tool exploration and the edit comes too late. Not a win at this budget.
- Both differences are within run-to-run noise (~±3), but neither shows a positive signal; not stacked.
- lb10_plan: lb04 + a zero-call text plan (PLAN block before the first call, a `[step | call | edited]`
  status line per message, re-plan after compaction, forced edit at call 12). Targets the empty-patch
  failure mode (paper arXiv 2609.20804: plan re-shown each turn, no-edit 68.6% -> 27.8%).

### lb10_plan result (2026-11-06 06:05 IST)

| variant | resolved | mean Kaggle-min | call-capped | time-capped | empty patch | mean calls | paired vs lb04 |
|---|---|---|---|---|---|---|---|
| lb04_public_7min | 16 | 5.48 | 14 | 15 | 11 | 22.3 | — |
| lb10_plan | 15 | 5.40 | 11 | 16 | 9 | 20.4 | 13 both, 3 only lb04, 2 only lb10 |

- The status line `[step | call | edited]` appears in all 39 traces: the agent follows the plan format.
- It moves the right way on the targeted mechanism (empty patches 11 -> 9, call-capped 14 -> 11, fewer calls,
  slightly faster) but the resolved count is a tie within noise (15 vs 16). First addition that is not worse
  on any process metric. Candidate for a public-LB check only with the user's OK.

- Submitted to Kaggle 2026-11-06 07:21 IST with user OK (ref 58041682). Public score pending.

### lb11_plan_early_edit result (2026-11-06 12:50 IST)

| variant | resolved | mean Kaggle-min | call-capped | time-capped | empty patch | loop steps (tasks) | paired vs lb04 |
|---|---|---|---|---|---|---|---|
| lb04_public_7min | 16 | 5.48 | 14 | 15 | 11 | 32 (21) | — |
| lb10_plan | 15 | 5.40 | 11 | 16 | 9 | 34 (21) | 13 both, 3/2 |
| lb11_plan_early_edit | 13 | 5.68 | 13 | 17 | 16 | 38 (23) | 12 both, 4 only lb04, 1 only lb11 |

- Moving the forced edit from call 12 to call 10 did not reduce empty patches; it raised them (9 -> 16) and
  loop steps rose too. Rushing the edit does not help when the time is lost in thinking loops. Dropped.
- lb12_loop_cap (lb10 + thinking 1024 / max_output 3072) started 12:49 IST (late: my completion watcher's
  command line matched the queue's pgrep pattern and blocked it for ~2 h; watchers now use '[o]vernight_screen').

- lb12_loop_cap dropped by user decision 2026-11-06 ~13:00 IST after 1 task; screen stopped, no result.

### lb13_locator result (2026-11-06 15:38 IST) - dropped

| variant | resolved | mean Kaggle-min | call-capped | time-capped | empty patch | mean calls |
|---|---|---|---|---|---|---|
| lb04 | 16/39 | 5.48 | 14 | 15 | 11 | 22.3 |
| lb10 | 15/39 | 5.40 | 11 | 16 | 9 | 20.4 |
| lb13 | 12/39 | 6.14 | 12 | 20 | 19 | 22.6 |

Paired vs lb04: both 12, only lb04 4 (rich_3777, rich_3676, rich_3521, rich_2943), only lb13 0.
Locator usage: called first in 37/39 tasks and returned a TARGET report in 33, but the sub-agent ignored
its "at most 6 tool calls" rule: mean 13.9 calls (range 4-28, 17 tasks >= 14); in rich_3470 it used all 28.
Sub-agent calls share the 28-call / 7-min budget, so the main agent was left too little to edit (empty
patches 11 -> 19). A prompt cap is not enforced; agent_tool exposes no per-sub-agent call limit. Any
retry would need a hard limit (e.g. a smaller max_output/thinking budget for the sub-agent, or a
locator LoRA trained on short trajectories). Not stacked; lb04 remains the best local variant.

## 2026-11-06 16:30 IST — LoRA data synthesis (user: "get more data or synthesise more", option b)
Inventory: 1,829 recorded runs, 505 resolved; on the 90 public tasks outside the 39-task screen
(kept held out): 252 resolved runs on 34 tasks, failures 317 touched a gold file, 158 wrong files,
262 empty; 56/90 tasks never solved (gold diffs median 24 lines vs 7 for solved; 10 need a new file).
Deviation from the plan: the September swebox traces store only thought *counts*, not thought
text, so they cannot train a thinking-on agent. Training data therefore comes from new official-
harness runs (traces keep the thinking text), graded with swebox (the official grader handles 41/129
public tasks; tools/regrade_swebox.py matched it 4/4 on lb04 patches). Splicing near misses (step 3)
is dropped: the official harness cannot resume a run mid-way; the hint passes cover those tasks.
- tools/gen_lora_data.sh (started 16:01 IST): lb04 at the screen budgets on the 90 tasks:
  plain -> hint1 (files of the gold fix) on the still-unsolved -> hint2 (+ symbols, size).
- tools/make_hinted_tasks.py writes data/lora/tasks_hint{1,2}.jsonl (hints_text; rendered by the
  harness as "## Hints:").
- tools/build_sft.py: resolved runs -> chat examples; hint section removed; repeated thought lines
  dropped, thought capped at 6k chars; hinted runs rejected if they edit before any search/read or
  mention the hint; <= 3 runs per task. Dry run on lb04+lb10 screen traces: 31 runs, 18 tasks,
  0-67 loop lines dropped per run.
- Step 4 (new repos): candidate nebius/SWE-rebench (filtered split, 33.9 MB parquet; real issue
  + test tasks from many Python repos); ingest = clone at base_commit + py3.13 wheels + selfcheck
  (gold resolves, empty fails). Waiting for the user's download approval.

### 2026-11-07 02:00 IST: LoRA data generation, pass 1 (plain) result; reboot recovery
- Windows rebooted at 21:52 IST on 11-06 and WSL would not start (E_FAIL, code 6): drive E: was full (0 GB),
  the ext4.vhdx (383 GB) could not grow. Moved the distro to <second drive> (wsl --manage --move,
  4.6 min). Inside Linux 376 GB is used, 201 GB of it Docker task images (197.6 GB unused; prune deferred
  to the user's review). All jobs resumed from their result files at 01:04 IST.
- Pass 1 (plain, lb04 bundle, 90 non-screen tasks): **30/90 solved** (swebox), 27 empty patches.
  Pass 2 (hint1, file list) runs on the 60 unsolved; the `## Hints:` section is in the prompt as intended.
- Rebench ingest: two fixes before the selfcheck ran: link sandbox/ and docker/ into the new data root,
  and write created_at as ISO with Z (install_deps compares it to tz-aware dates). Commit e294040.

### 2026-11-07 02:10 IST: SWE-rebench ingest result
- Selfcheck on 319 snapshotted tasks: gold resolves 144, empty resolves 2 -> **kept 142 tasks from 116
  repos** (data/external/rebench_root/competition/tasks.jsonl; the full 320 in tasks_all.jsonl).
- Gold failures (175): exit 2/4 (collection) 133, exit 1 (tests fail) 41, 1 not applied. Collection
  errors are a long tail of environment gaps: undeclared imports (requests, mock, jwt, zmq, lxml),
  unbuilt C extensions / generated _version modules, pydantic_core version clashes, pkg_resources.
  Not rescued: each needs a per-task fix and 142 tasks are enough for the first rebench pass.
  Possible later: install the SWE-rebench `requirements` freeze verbatim instead of declared deps.

### 2026-11-07 05:10 IST: pass 2 (hint1) result; hint-sentence redaction in build_sft
- Pass 2 (file-list hint) on the 60 unsolved: **9 solved**, 12 empty patches. Pass 3 (hint2) runs on 51.
- All 9 hinted solves cited the hint in their thinking ("The hint says the fix is in X."), so the
  run-level rejection dropped every one. Changed tools/build_sft.py to drop only the sentences that cite
  the hint (1-3 per run) and keep the rest; "type hint" is no longer a match. SFT probe: 39 runs from
  39 tasks (30 plain + 9 hinted), none mention hints/maintainer, user prompts carry no Hints section.
- Known residue: after redaction the model sometimes names the right file before reading anything
  (e.g. "Let's start by examining rich/ansi.py"). Accepted for v1; the screen will show if it hurts.

### 2026-11-07 08:45 IST: LoRA trainer on the int4 checkpoint (smoke test)
- Pass 3 (hint2): 2 solved. Dev-task generation total: **41/90 tasks solved** (30 + 9 + 2); sft_v1 = 40 runs.
- Deviation: PEFT cannot wrap the checkpoint's Linears (compressed-tensors pack-quantized: weight_packed /
  weight_scale, no .weight), and transformers cannot even run their forward as loaded. New tools/qlora.py:
  every packed Linear (410, all in the language model) becomes a QLinear that rebuilds its bf16 weight
  from int4 (symmetric, group 32) in forward and again in backward (custom autograd function, nothing
  saved); LoRA (fp32 master, bf16 compute) on q/k/v/o = 230 modules, 45.0M params. Unpack verified
  bit-exact against compressed_tensors.unpack_from_int32; outputs and grads exact against a reference.
  Adapter saved in PEFT key format for vLLM.
- Sanity: loss of the untrained model on its own solved runs is 0.25-0.76 (a wrong dequant would be ~10+).
- Memory (RTX 5090, 32 GB; weights 19.5 GiB): past VRAM the WSL driver spills to system RAM (one 12k step
  took 1219 s), so the trainer now caps PyTorch at 95% of VRAM (OOM instead of spill). expandable_segments
  breaks on WSL (no CUDA VMM: "CUDA driver error: unknown error").
  8k on GPU: OOM. **8k with --cpu-offload (saved activations in pinned RAM): 32-38 s/step, peak 25.6 GiB.**
  12k with offload: one step 79 s / 28.1 GiB, then OOM. -> train at 8k with offload (~3.2 h for 2 epochs).
- Cost of 8k windows: target tokens 148k vs 273k at 12k (the system+tools+task head and long tool
  outputs leave too little room, and windows that cannot fit one turn are dropped). To fix in windows().
- 09:10 IST windows() fix: when the overlap turns leave no room, retry without them; when one turn
  alone exceeds max_len, skip that turn and keep the rest of the run (it used to drop the rest). Sizes:
  head (system+tools+task) median 4377 / max 7266 tokens; runs median 22k tokens. At 8k: 288 windows,
  2.14M tokens, **272.8k target tokens (was 148k; 12k gives 272.6k)**. Spans checked by decoding: a target
  starts after `<|channel>thought\n` and ends with the `<|tool_response>` stop token. ~5.6 h for 2 epochs.
- 17:00 IST rebench generation, plain pass: **56/142 solved (swebox)**, 39% (dev plain pass was 30/90).
  3622 model requests recorded. Pass took 8h20m (~3.5 min/task). hint1 started on the 86 unsolved.
- 22:24 IST rebench hint1 pass (gold file list): **13/86 solved** (dev hint1 was 9/60). Rebench so far
  69/142; with dev, 110 solved tasks. hint2 started on the 73 still unsolved (~03:00 IST).

### 2026-11-08 00:15 IST: LoRA v1 configuration (one training run, approved by the user)
Literature check (user asked to tune once): LoRA Without Regret (Thinking Machines 2025), SWE-Gym, SWE-smith,
R2E-Gym, Hindsight Hint Distillation (HHD, 2605.11556), SWE-Lego (2601.01426), Step Rejection Fine-Tuning
(2605.10674), trajectory curation for LoRA (2607.17205), Unsloth Gemma 4 guide.
- Deviation: LoRA on all Linears (q/k/v/o + gate/up/down; 410 modules), not attention only: attention-only
  underperforms even at matched parameter count (LoRA Without Regret; QLoRA; Unsloth default).
- lr 1e-4 -> 2e-4 (LoRA optimum ~10x full-FT; SWE full-FT uses 1e-5..5e-5; SWE-Gym 32B LoRA 5e-4). r16, alpha 32.
- New: error-step masking (SWE-Lego +2 pts; SRFT +1.3 over RFT): a model turn whose tool call failed as a
  call (bad args, edit target not found, write outside workspace; not CommandError = a command that ran and
  exited non-zero) stays in context with no loss. Dev runs: 28/973 turns masked, target tokens 273k -> 258k.
- 5 runs held out for a validation loss; adapter saved after each of 2 epochs (both get screened).
- Risk noted: SWE-Gym's 32B trained on its own successes got worse (15.3 -> 8.7%); plain self-RFT gives
  +0..1 pt elsewhere. HHD (hinted on-policy rollouts, like our hint passes) gave +8 on Verified.
- Order (tools/lora_v1_pipeline.sh): wait for generation -> vLLM 0.19.1 canary with a rank-16 all-Linear
  adapter (attention lora_B = 0, MLP lora_B random: output must change; also decode speed) -> attention-only
  if vLLM cannot serve it -> build sft_v2 (all 6 passes) -> 8k smoke -> train.
- 03:11 IST rebench hint2 pass (files + symbols + size): **9/73 solved** (dev hint2 was 2/51). Rebench total
  **78/142** (56 + 13 + 9). With dev (41): 119 solved tasks for sft_v2. tools/lora_v1_pipeline.sh takes over.
- 03:16-09:40 IST local training failures, all on memory (RTX 5090 under WSL), fixed in train_lora.py:
  (1) the logits loss ran inside save_on_cpu, so each chunk copied the 2.6 GiB tied embedding to pinned RAM
  (OOM at step 3) -> loss outside the offload; (2) cache fragmentation (5 GiB reserved but unusable,
  OOM at step 2) -> empty_cache when >2 GiB is unused, loss chunk 256; (3) AdamW state for 122M params
  (1 GiB) on the GPU made every step after the first OOM -> optimizer on CPU with master copies;
  (4) the pinned host cache grew 0 -> 8 -> 12 GiB ("shared GPU memory") -> released every window.
  Also: resumable state every 10 steps, OOM steps skipped instead of crashing. Local speed after the
  fixes: ~50 s/window (~10 h/epoch).
- Deviation (user request): training moved to a Modal H100 80 GB (free Starter credit, $30/month;
  tools/modal_train.py). Same code, data, seed and settings, no CPU offload. The checkpoint upload was
  instant (Modal already had the identical public file). Identical start: initial loss 0.2452, val 0.3829,
  step-1 loss 0.3915 on both machines. Speed ~118 s/step (8 windows) = ~6 h for 2 epochs (~$24).
  Local run stopped at step 6 (user).
- 12:40 IST epoch 1 on Modal: held-out val loss 0.3829 -> 0.2767 (5 runs). Mean train loss per 10 steps
  0.32 -> ~0.22.
- 13:20 IST the Modal run was cancelled at step 112/181 (not by the credit guard, spend ~$15.5): the
  launcher blocked on train.remote() and the local client hit a network timeout ("Deadline exceeded"),
  which cancelled the function. Fix: new `start` entrypoint that spawns the function (modal run --detach
  ...::start), so the run no longer depends on the local client. Resumed 13:25 IST from state.pt at step
  110 (new app ap-cPsWmfoHq7GAX6vX16PYqv); steps 111-112 reproduced the original losses
  (0.2177/0.2175, 0.2212/0.2220). Cost of the incident: ~2 steps + reload, ~$0.6.
- 16:12 IST LoRA v1 training done on Modal (exit 0; resumed segment 2.71 h). Held-out val loss:
  before 0.3829, epoch 1 0.2767, epoch 2 0.2729 -> epoch 2 screened (user rule: best val epoch).
  Total Modal spend $26.58 metered (cap $29.50). Screen of lb04_lora_v1_e2 on the 39 gradable tasks
  started 16:09 IST (time scale 0.625, local vLLM 0.19.1 with the adapter).
- 17:25 IST LoRA screen INVALID, restarted. 0/19 resolved, but 15/19 test logs were pytest crashing at
  start-up (pluggy PluginValidationError, plugin 'flake8'); rich_3882 had a patch byte-identical to
  lb04's resolved one. Cause: swegemma caches the unpacked wheel pool at
  <tempdir>/swegemma_sp_cache_v8/sp_base.tar keyed only by name, not by wheels dir. The rebench
  generation (1450-wheel pool incl. pytest-flake8 for wemake-python-styleguide) built it 11-07 03:02
  IST, and every later public-pool run reused it. Fix in tools/official_screen.py: TMPDIR =
  /tmp/swegemma_<sha1(COMP_DIR)> before swegemma is imported (one cache per pool). Old cache kept as
  /tmp/swegemma_sp_cache_v8.rebench_pool_1107; invalid results kept as
  data/screen_lora/lb04_lora_v1_e2.INVALID_flake8_env. Affected nothing else: dev generation ended
  02:21 IST (before the cache), rebench passes used the matching pool, lb04 base screen predates it.
  Restarted 17:31 IST: no plugin errors, tests run (rich_4077 resolved, 4 passed).
- 19:35 IST LoRA v1 screen stopped at 32/39 (user). Adapter 9/32 vs lb04 12/32 on the same tasks;
  gained none, lost rich_3777 (no patch left; tests ran on the unmodified repo), rich_3676 (notes
  stored as safe_str(list) instead of the list -> 1 test fails), rich_3521 (patched the symptom in
  split_cells instead of rewriting the cut search -> 4 fail). Behaviour barely changed: tool calls
  20.3 vs 21.6/task, time between tool calls +15% (matches the -14% adapter decode), thoughts mean
  920 vs 797 chars; tasks with no edit at all 9/33 vs 5/39; time-outs 12 vs 12 but the adapter's
  time-outs mostly end with no patch (9 vs 4). Verdict: LoRA v1 (rejection-sampled own runs, val loss
  0.383 -> 0.273) does not beat lb04; not submitted. lb04 stays the best bundle.

### 2026-11-08 19:55 IST: timeout analysis -> lb14_temp1 (user picked)
Where lb04's time goes (39 traces): timed-out tasks (12) have a median longest turn of 4,130 output
tokens (at the 4,096 thinking cap) vs 968 in resolved ones; 1.8 turns >1,500 tokens per task; the 3
slowest turns take 62% of the task's time; first edit at 195 s median (5 never edit). Of the 18 turns
>=3,000 tokens, 8 are repetition loops (>25% duplicate sentences, e.g. rich_3105 223/241), the rest
long deliberation. Across screens: lb04 (temp 0.2) 0.46 such turns/task, 8 loops; lb07 (temp 1.0,
other changes) 0.23/task, 2 loops; lb05a (thinking 2048) ~0 but 12/39 (deliberation is useful).
lb14_temp1 = lb04 with sampling only: temperature 1.0, top_p 0.95, top_k 64 (Gemma's recommended),
thinking budget unchanged. Screen started 19:55 IST, same 39 tasks, time x0.625.
- 20:20 IST lb14_temp1 on the first 5 of lb04's 12 timed-out tasks (user: 5 only): 0/5 resolved.
  Loops gone (0 repetition loops, longest turn median ~2,000 tokens vs ~4,100), time-outs 5 -> 2, every
  task now reaches an edit, but 3/5 instead hit the 28-call cap and all patches fail tests
  (rich_3953 1 failed, rich_3935 2, rich_3942 6, rich_3934 1; fastapi_15661 collection error).
  Reading: on these tasks the loops are a symptom, not the blocker; removing them turns time-outs into
  call-cap failures with wrong fixes. Not extended to the full 39. Parked.
- Kaggle (checked 2026-11-08 20:40 IST): lb10_plan (ref 58041682) public 0.10 vs lb04 0.15. Locally
  15/39 vs 16/39 with better process metrics, yet lower on the board: another sign the rich-heavy
  39-task screen does not rank variants the way the hidden set does. lb04 remains the best public score.

### 2026-11-08 21:00 IST: Kaggle check (leaderboard, notebooks, forum)
- Board: top 0.29 (Yurnero, since 10-29); 11 teams at 0.24 = honghanhhh "Gemma 4 Baseline | LB 0.24"
  notebook, which is lb04 with 8 min instead of 7 (our notes: those bytes scored 0.17-0.24 across
  public runs). Us 0.15, rank 2878/5105. Deadlines: main 12-02, paper 11-12; team-merge ~11-25.
- Forum 755677 (Kapil Chopra): a run that goes past the 12 h total ends "complete, no score" (their
  adapter at 8 min/4096 went over; 6 min/2048 scored 0.17). Confirms keeping lb04 at 7 min.
  Also: LoRA on this base does load on the scorer (Yurnero, Ankith Savio); adapter costs ~29% time.
- Forum 753973: vLLM 0.19.1 gemma4 tool parser can loop forever on an ARRAY argument holding a bare
  element with ']' (e.g. ["Optional[int]"]); it freezes the server, every later task scores 0 and the
  run can exceed 12 h. No host answer on a fix. lb04 still lists get_code_subgraph(nodes: list[str])
  (prompt forbids it; 0 graph calls in 39 local tasks, but ~120 hidden tasks). Removing the three graph
  tools from agent.yaml removes the only array-argument tool at no cost.
- Forum 754783: compaction summarizer ignores tool calls/results (text parts only); host will not move
  to ADK v2. Matches our earlier "compaction drops edited-file content" finding.
- Forum 744272 (double-JSON tool results): fixed in the 09-30 wheelhouse (we run V31).
- 21:00 IST lb15_no_graph submitted with user OK (ref 58132834): lb04 (7 min, temp 0.2, thinking 4096,
  28 calls) with get_code_neighbors / search_similar_code / get_code_subgraph removed from agent.yaml and
  the prompt line naming them removed. Nothing else changed. Official compile check ok; zip sha256
  a94dee1e...a1c6 (data/submissions/lb15_no_graph). Purpose: remove the only array-argument tool
  (parser-freeze risk) at no expected cost (0 graph calls in lb04's 39 local tasks). Score pending.

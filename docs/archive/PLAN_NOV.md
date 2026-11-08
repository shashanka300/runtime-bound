# Plan, 2026-11-04 → 2026-12-02 (replaces the open items in NEXT.md)

## Situation
- Deadlines: **Paper Track writeups Nov 12** (8 days); entry/team merge Nov 25; **final
  submissions Dec 2**. Two final submissions per team; the E2B/E4B bonus ($10k) must be one of
  them.
- Board: #1 0.29, several 0.24; public configs 0.18–0.20 (31B/12B, max_output 8192, thinking
  budget 4096, 8 min/task, 240 s commands). We are at 0.06 with two September submissions.
- Scorer changes since September (current wheelhouse, "V31"):
  - LoRA adapters load and score (forum + #1 uses one).
  - Earlier thoughts are kept between tool calls (the dropped-`reasoning_content` bug we found
    by replay, independently reported in 744354, fixed by the host).
  - `thinking_budget` and `seed` reach vLLM (744566).
  - Gemma 4 12B, E4B and E2B allowed (one variant per submission).
- Unchanged: 1 submission/day, one pending at a time, 14–33 h queue; 12 h run limit for ~120
  sequential tasks; ParallelAgent+tools broken in ADK 1.36; hidden tasks are from other (non-async)
  repos, so public-task scores are optimistic and graph tools cover every hidden task.
- Assets: 1,438 recorded agent runs (375 resolved), fidelity tooling (swebox official modes,
  vllm019 env, official checker), one RTX 5090.

## Track A — Paper (Nov 4–12), first priority
Working title: **"Runtime-bound: what limits a local 31B coding agent, and what local evaluation
gets wrong"**. Contributions, all already measured:
1. Accounting (E1/E2): 90% of starter failures are context exhaustion, caused by reserving
   16k of the 32k window for output; one config line +12.5 pts (p = 0.006).
2. Architecture (E5): phase decomposition with fresh contexts fails G3; the loop cause
   (near-greedy sampling in low-thinking phases) is fixed by sampling (17 → 0 loops), yet the
   phase agent still loses under realistic budgets (negative result).
3. Fidelity ladder: a local-vs-scorer decomposition that surfaced three scorer defects — budget
   never sent, earlier thoughts dropped, LoRA wiped — all later confirmed or fixed by the hosts;
   plus the stall artefact we caught and corrected (KV accounting at concurrency 2), and the
   time-budget curve.
4. Public dev-set reliability: released tooling grades ~41/129 public tasks offline (missing
   test deps, py3.13 SSL, installed-package shadowing, wheel-cache bug), independently confirmed.
5. Consumer-hardware telemetry (energy/latency per task) from day one.
Daily: Nov 4–5 outline + figure list; Nov 6–8 draft sections 1–4; Nov 9 post-fix validation
results in (A2); Nov 10 full draft + numbers audit (every number traced to a run); Nov 11
polish; **Nov 12 submit**. Second writeup (Best New Resource: the evaluation suite) only if the
main draft is done by Nov 10 — the text is largely reusable.

A2 — one post-fix validation for the paper (GPU, unattended, ~5 h): on the V31 scorer engine at
L4 speed, concurrency 1: arm G (thinking off) vs arm F' (thinking on, budget 4096, max_output
8192). Answers "with thoughts kept and budgets honoured, does thinking help?" and closes the
ladder section with the fixed engine.

## Track B — Leaderboard (Nov 4–Dec 2), runs in parallel on the GPU and the daily slot
B0 (today): update to wheelhouse V31 locally (vllm019 + official env), re-run the official
checker and the thinking cross-check, pin hashes in the design log.
B1 (first slot): **lb04 = public-config calibration** on the fixed scorer: 31B, no LoRA,
thinking on (budget 4096, max_output 8192), 8 min/task, 240 s commands, plus our prompt hygiene
(no `rg`, `tail` long outputs, ranged reads under the 150-line cap). Expect ~0.18–0.20; tells us
where we stand with the fixes.
B2: model-size sweep on the scorer engine at L4 speed (local, unattended): 31B vs 12B with the
same agent; choose the main line for the remaining slots.
B3 (Nov 13–24, after the paper): one lever at a time, each a submission:
  - LoRA SFT/RFT on our 375 resolved trajectories (and new ones), on 12B first (fits the 5090
    comfortably); 31B QLoRA only if 12B gains;
  - prompt/efficiency from the runtime audits (never restate files, ranged reads, early submit);
  - graph tools for localisation (hidden repos have full graph coverage), tested on rich tasks.
B4 (Nov 25–Dec 2): pick the two finals (best main line + best E4B/E2B); no new ideas after Nov 28.

## Track W — Weight edit of an agent behaviour (paper section; added 2026-11-04 night)
Idea: weight-space programming (MIT CSAIL blog, Takehi et al.; replication on Gemma 4 12B by
ValerioDolci/frontier-weight-edit with Claude Opus 5.5 as editor). Paper framing: an
intervention ladder on one budget-wasting behaviour (prompt rule -> budget change ->
diff-of-means ablation -> frontier-agent weight edit), measured on behaviour rate AND task success
under the scorer budget. A null result is reportable ("edits cannot buy back the budget").
Decisions (user, 2026-11-04): model Gemma 4 12B; up to 2 headless Opus 5.5 editor sessions
(second only if the first fails the behaviour check); an edit that wins locally is a normal
leaderboard candidate (note the closed-model-editor question in the submission description).
W1 (Nov 5 am, CPU) target behaviour from V1/V3 traces; GO only if it occurs in >= 25% of failed
   tasks. Pre-register the prediction before any edit.
W2 decision-point set: ~100 points before the behaviour + ~100 controls (offline next-action rate).
W3 (Nov 5 pm) harness: CPU kernel over 12B weights, forward-pass budget of 50 (teacher-forced
   activations / generations), replication protocol.
W4 (Nov 6) arms: diff-of-means ablation, editor session(s); convert the edit to a rank-8 LoRA
   against the dequantised int4 base; verify on vLLM 0.19.1 with tools/lora_canary.py; official check.
W5 (Nov 6 night) end-to-end on the 39 screen tasks: edited 12B vs base 12B (lb06_12b).
W6 (Nov 7-12) write the section.

## Track C — Small-model bonus (E4B/E2B), from Nov 13
E4B is fast enough to iterate quickly on one GPU: same agent + LoRA on E4B; one final slot.

## Rules for the next 4 weeks
- Paper work never waits on the GPU; GPU runs are queued to run unattended overnight.
- Every submission is pre-checked with leaderboard/official_check.py on V31.
- Local public-task numbers are directional only (other repos on the board); decisions on the
  main line use the leaderboard.
- Ask before each multi-hour run, as before; daily status in IST.

# E1b — Context-budget fixes for the starter agent (31B)

**Why (inserted 2026-09-26, see docs/design_changes.md).** E1/E2 found 93.5% of starter
failures are context exhaustion. vLLM rejects a request when prompt + `max_output_tokens`
(16384) > 32768, so the starter's usable prompt is 16k, and runs die while still localizing.
Before building the phase-decomposed agent (C2), this measures what cheap fixes recover.

| Arm | Change (vs parent) | Submittable |
|---|---|---|
| A | starter as-is (E1's dev40 × seeds 0,1 runs reused: identical config and server) | yes |
| B | A + `max_output_tokens = thinking (4096) + 4096 = 8192`: usable prompt 16k → 24k | yes |
| C | B + prompt guidance (`src/arms.py:GUIDANCE`): grep then ≤80-line reads; scratch via `run_command` in `/tmp` | yes |
| D | B + up to 2 fresh-context retries on the same workspace after a context death (`Budget.context_retries`) | no: diagnostic of what recovering context deaths is worth |

**Pre-registered comparisons:** B−A (reserve fix), C−B (guidance), D−B (recovery), C−A
(best submittable vs starter): paired on task × seed, 95% bootstrap CI stratified by repo, exact
McNemar. A fix counts only if its CI excludes 0. **Exit:** the best submittable arm becomes
the first leaderboard submission and the base for E3/E5.

**Design:** dev40 × seeds {0, 1}, 6 min per task, concurrency 2, starter LoRA adapters served
in every arm. 3 new arms × 80 = 240 sessions (~6 h). Tests: `tests/test_arms.py`.

```bash
bash tools/run_notebook.sh E1b_context_fixes main '{"params.reuse_arm_a": "<E1 run>/artifacts/swebox/t6min"}'
```

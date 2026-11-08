# E5 — Phase-decomposed context architecture (C2, gate G3)

**H4 / G3 (pre-registered).** A phase-decomposed agent raises the resolved rate by ≥ 5 points
over the tuned monolithic agent at equal wall-clock; falsified if the paired 95% CI of the gain
includes 0. **Baseline:** E1b arm B (starter + output-reserve fix), same dev40 × seeds {0, 1},
same 6-min budget, model, sampling and adapters.

## Why this design (from E1/E2/E1b evidence)
- 90% of starter failures are context exhaustion, mostly *during localization* (E2/H3).
- The reserve fix alone takes dev40 from 18.8% to 31.2% (E1b), but one context still has to
  carry exploration, editing and verification.
- The largest occupancy cost is tool output that stays in context for the rest of the task:
  truncated reads 21% of all processed tokens, stale thinking 16% of prompt tokens (E2).

## Agent (`src/build_submission.py` → declarative YAML, submission-legal)
```
SequentialAgent phase_agent
 ├─ localizer_agent   read-only tools            → state.localization (files, root cause, fix plan)
 ├─ reproducer_agent  run_command (scratch /tmp) → state.repro (command, current vs expected)
 └─ LoopAgent fix_loop (max 3)
     ├─ editor_agent    read/edit tools, sees localization + repro + verification? → state.edit_summary
     └─ verifier_agent  runs repro + targeted tests; pass → clean scratch, submit_patch
                        fail → state.verification (fed to the next editor iteration)
```
- `include_contents: none` on every phase: per ADK (`contents._get_current_turn_contents`) a
  phase sees only the latest user/other-agent message plus its own tool calls. Raw tool
  output never crosses a phase boundary.
- The issue reaches every phase through `{problem_description}` in the instruction. Prompts are
  generated as `shared.md + <phase>.md`, so all phases share a byte-identical prefix
  (tested: `tests/test_submission.py`).
- The shared rules encode E2/E1b findings: small targeted reads, scratch via `run_command` in
  `/tmp` (the `write_file` trap), and never the full test suite.
- If the pipeline ends without a submission, the harness nudge re-runs it with fresh contexts;
  session state (localization etc.) persists.

## How to run
```bash
bash tools/run_notebook.sh E5_context_architecture v1 \
  '{"params.baseline_dir": "<E1b run>/artifacts/swebox/arm_B"}'
uv run python -m src.build_submission /tmp/phase_v1   # inspect the generated submission
```

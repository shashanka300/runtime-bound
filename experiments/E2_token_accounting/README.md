# E2 — Where do the tokens go? (H2)

**H2 (pre-registered).** Over 40% of tokens are re-reads, re-prefill, or truncated tool
output; falsified if the waste share is under 20%.
**Design:** offline replay of swebox step traces (no GPU). **Budget:** minutes of CPU.
**Decision rule:** supported if the lower bound's 95% CI > 40%; falsified if the upper
bound's CI < 20%; otherwise inconclusive.

## Method (`src/accounting.py`)
Per model call, two views:
- **Compute:** prompt = cached (prefix-cache hit) + new (appended since the previous call in
  the same context) + **recomputed** (prefix the cache lost: KV eviction or compaction);
  decode = thinking + output.
- **Occupancy:** request characters by kind converted to tokens with a per-agent linear fit
  (`tables/e2_char_token_fit.csv`, R² ≈ 0.97). Tool outputs are segments that stay in the
  context for every later call; flagged **re-read** (≥ 50% of the lines already returned in
  the task, by `read_file` ranges or `cat`/`sed -n`/`head`/`tail`), **truncated**
  (`read_file` hit 150 lines / 10k chars, or `run_command` > 5000 chars) or **oversized**.

Waste = re-read + truncated (token × calls in context) + recomputed prefill, as a share of
all processed tokens. Categories can overlap, so the sum is the **upper bound** and the largest
single category the **lower bound**. CIs: bootstrap over task runs, stratified by repo.
Tests: `tests/test_accounting.py` (hand-computed trace).

## Inputs
`params.input_dirs`: swebox results directories (e.g. an E1 run's `artifacts/swebox/t6min`).
Empty: the notebook first produces a small run itself, so E2 never depends on E1 having run.
```bash
bash tools/run_notebook.sh E2_token_accounting label '{"params.input_dirs": ["<dir>"]}'
```

## Results: E1 6-min traces, partial (164 task runs, 2026-09-25)
**H2: INCONCLUSIVE**. Waste share 24.4% [22.1, 26.7] (lower) to 33.7% [30.5, 37.0] (upper).

| Category (share of processed tokens) | Estimate |
|---|---|
| truncated tool output (in context) | 20.5% [18.3, 22.7] |
| recomputed prefill (cache misses) | 11.1% [9.1, 13.3] |
| re-reads | 2.1% [1.5, 2.7] |

Compute view: 78% of processed tokens are prefix-cache hits, 8.5% new prefill, 10.8%
recomputed, 1.8% thinking, 0.7% output. Occupancy: instructions + task 35%, **stale thinking
re-sent 16%** (not pre-registered; ADK re-sends past thinking text), tool output 43% (half of it
truncated reads).

### Finding: the starter can use only half the context window
Every one of the 124 rejected requests carries vLLM's error *"you requested 16384 output tokens
and your prompt contains at least 16385 input tokens"*. vLLM rejects a request when
prompt + `max_tokens` > 32,768, and the starter sets `max_output_tokens: 16384`, so its usable
prompt is 16k tokens. The largest real output in the traces is a few thousand tokens. The same
configuration runs on the official box, so this is not a local artefact. `figures/e2_context_growth.pdf`
shows contexts reaching this 16k line, not 32k. Lowering `max_output_tokens` is the cheapest
first intervention for C2 / E5.

## H3 — failure taxonomy (gate G2) (`src/taxonomy.py`)
**H3 (pre-registered).** Context-loss failures are a top-3 failure mode; falsified if
mislocalization and wrong edits dominate. **Rule:** supported if `context_exhausted` ranks in
the top 3 primary modes of failed runs on gradable tasks.

Primary label, first matching rule (proximate cause): `context_exhausted` → `time_exhausted` →
`gave_up_no_patch` → `patch_apply_failed` → `mislocalized_file` → `mislocalized_region` →
`wrong_edit` → `other`. Localization compares the agent's source-file hunks with the reference
patch (±5 lines). New files the reference does not create are **scratch**, not localization
attempts (see the `write_file` finding below). A second, patch-only label ignores how the session
ended. Markers: repeated identical actions, re-reads, reverted edits, whether a reference file
was viewed, scratch files in the patch. `tables/e2_failure_spotcheck.csv` holds a random
sample with evidence for human verification. Tests: `tests/test_taxonomy.py`.

### Results: E1 6-min traces, partial (204 runs on gradable tasks, 153 failed)
**H3: SUPPORTED. G2 passes.** `context_exhausted` is the #1 mode: **93.5% [89.5, 97.4]** of
failures. Wrong edits 3.9%, time 2.0%, mislocalized 0.7%.

- **Context kills runs before they reach a fix.** Of 143 context-exhausted runs, 115 had no
  source edit yet, 18 had edited a non-reference file, 8 had edited the reference region.
- **It is exhaustion, not confusion.** Failed runs viewed a reference file in 62% of cases
  (resolved: 96%). Repeated actions (18% vs 45%) and re-reads (43% vs 45%) are *not* more
  common in failures. The classic forgetful-loop symptoms do not explain the failures; running
  out of the 16k usable prompt during localization does.
- **`write_file` path trap (spec-level).** `write_file` strips a leading "/", so
  `write_file("/tmp/repro.py")` writes `/workspace/tmp/repro.py`, which enters the patch. The
  prompt tells agents to put scratch files in `/tmp`, but the tool cannot do so. 67% of
  *resolved* patches carry such files (harmless to grading here, but they pollute submissions).

## Outputs
`tables/e2_h2_test.csv`, `e2_compute_view.csv`, `e2_occupancy_view.csv`,
`e2_char_token_fit.csv`; `figures/e2_token_accounting.pdf` (compute, occupancy, H2
distribution), `e2_context_growth.pdf`, `e2_char_token_fit.pdf`.
Final numbers are produced when E1 (6 + 10 min × 3 seeds) completes.

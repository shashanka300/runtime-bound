# Track W, session 1: FINAL

## Submitted edit: none (no edit)

None of the six edits I evaluated reduced redundant re-fetching (`reread+dup+regrep`) beyond noise
without control damage. I stopped after 6 of 50 evaluations because the routes these constraints
allow are exhausted (reasoning below), not because the budget ran out. The deliverable is a
documented negative result plus the mechanistic findings that explain it.

## Dev results (60 target + 60 control points, temperature 0.2, seed 0)

| edit | what it does | target_rate | ctl_flagged | ctl_tool_changed | no_tool_call | verdict |
|---|---|---|---|---|---|---|
| base | none | 0.733 | 0.167 | 0.233 | 0.008 | reference |
| (pre-session near-null) | random rank-1, scale 1e-4, L30 | 0.700 | 0.117 | 0.167 | 0.008 | null band ~0.68-0.70 |
| e01_r40abl_mlp29-40 | remove "Wait-mode" redundancy direction (r40) from MLP writes, L29-40 | 0.700 | 0.200 | 0.233 | 0.000 | null |
| e02_e40refl | reflect commit direction (e40) in MLP 35/39/40 and attn 37-40 | 0.700 | 0.200 | 0.250 | 0.017 | null |
| e03_readtok_abl | ablate raw `read` token readout from L44-47 MLP+attn writes | 0.683 | 0.483 | 0.833 | 0.850 | broken |
| e04_actadd_e40 | ActAdd of commit direction, +2.5 sd at L40 (MLP 34-40) | 0.683 | 0.150 | 0.233 | 0.008 | null |
| e05_actadd_e40_x4 | same, +10.8 sd (MLP 32-40) | 0.550 | 0.183 | 0.567 | 0.292 | degraded, non-selective |
| e06_nowait_supp | one programmed L47 neuron: one-sided penalty on "Wait" | 0.700 | 0.200 | 0.267 | 0.050 | null |

Any mounted adapter seems to land at 0.68-0.70 (the pre-session near-null edit gave 0.700). With
n = 60, nothing below about 0.60 that leaves the controls intact was ever reached. E05's 0.550
comes mostly from 22 targets producing no tool call, plus garbled arguments
(`rich/traceback.,py`). Its controls changed tool 57% of the time.

## Why the behaviour cannot be removed this way (mechanistic findings)

1. **Most of the target set is not something the model can see.** Of the 60 dev targets, 43 come
   after context compaction. Checked against only the calls visible in each prompt, 32/60 are
   not redundant from the model's point of view: compaction summarised away the earlier read, and
   the base model re-does these 27/32 times. To edit after compaction the agent has to re-read the
   exact lines, because `edit_file` needs `old_string`. Even a perfect fix of in-context redundancy
   would therefore leave target_rate around 0.45, and an edit that suppressed these reads would
   hurt the agent. The other 28 are genuine in-context loops or overlaps, which the base repeats
   about 61% of the time. Examples: the same `grep -n "render_renderables"` issued 6 times while
   the thought says "Wait, I already did that"; and boundary overlaps such as reading 1299-1350
   and then 1350-1450.

2. **The action is decided inside the thought, where there are no activations.** The chat
   template re-renders every earlier reasoning block, so loops are self-copying ("Let me search
   for X again"). The provided "decision" token is the `\n` after `<|channel>thought`, i.e. the
   first token of the new thought.
   - Directions found there are readable: a redundancy direction whose logit lens reads
     "Wait/Yes/So/Anyway" against "now/let/try"; a commit direction, "now/OK/perfect" against
     "search/explore". Leave-one-task-out probe AUCs are only about 0.65-0.76.
   - These directions are correlates, not levers. Ablation (E01), reflection (E02) and ActAdd at
     2.5 sd (E04) changed nothing. At 10.8 sd (E05) the model degrades before it steers, and the
     steering it does is indiscriminate.
   - The base re-run already flips 16 of 60 targets from the identical decision state, so that
     state only sets the odds.

3. **Late-layer token edits are powerful but brittle; lexical levers don't change plans.**
   - γ_final reaches 604, so token readouts share a large common component. Ablating the raw
     `read` readout (E03) broke decoding: 85% of outputs had no tool call, and `<image|>` tokens
     appeared inside arguments.
   - I added an offline screen to catch this before spending evals. It recomputes the
     first-token distribution at the 614 decision tokens from the provided L47 activations
     (direct path, final norm and unembedding only; no forward pass). It flags E03 in hindsight.
   - With it I built a one-neuron "programmed suppressor" at L47. Gate = up reads a contrastive
     readout of the token, so it fires quadratically only when the token is about to be emitted;
     down writes against it. This is a clean asymmetric logit penalty: P(Wait as next token) went
     from 0.117 to 0.026, median KL 0.016.
   - Removing "Wait" did not change *what* the model fetches (E06 null). The plan sentence
     survives without it.

4. **Caution on the metric.** `classify()` only tracks `read_file` ranges and `grep/rg/find`
   patterns. An edit that pushes the tool choice away from `read_file` would mostly re-route the
   same re-reads into `sed -n` / `cat` / `head`, which the flagger cannot see. target_rate would
   drop without freeing a single call; E05's outputs already show reads turning into shell reads.
   I deliberately did not try such tool-substitution edits, and I'd suggest screening future
   submitted edits for this.

## What I tried that failed (summary)
- Projection ablation and reflection of decision-token directions: redundancy "Wait-mode" r40
  (MLP L29-40) and commit e40 (MLP and attention L35-40). Null.
- Activation addition of the commit direction through rank-1 down_proj deltas that read the
  consistent decision-token activation pattern (L34-40 / L32-40). Null at 2.5 sd, degraded at
  10.8 sd.
- Raw token-readout ablation at L44-47 (`read`). Broke decoding.
- Projection-based "NoWait" variants, screened offline only: unreliable. They can only
  remove or flip existing components, and one variant *raised* P(Wait).
- Programmed suppressor neuron for "Wait" (NoWait). Clean offline, null online.

## What would be needed instead
- **Harness level:** this is what would actually free calls.
  - Keep the content (or the exact line ranges) of files being edited through compaction, or put
    them in the summary.
  - Add a duplicate-call guard that answers a repeated call with "already run; result was ...".
  - Note that budget warnings ("Only 0 tool call(s) remaining ... Finalize your edits") are
    currently ignored by the model.
- **For a future weight-edit session:** provide activations at in-thought positions, at least
  the last plan sentence and the tool-name token after `call:`. Selective edits would need them.

## Files
- Running log: `<session dir>\NOTES.md`
- Per-eval summaries: `<session dir>\results.jsonl`
- Edits (deltas + LoRA): `<session dir>\edits\`
- Offline screen and programmed suppressor scripts: `ocheck.py`, `supp.py`; ActAdd builder:
  `e04_build.py`; projection builder: `mk.py` (all in `<session dir>\`)

# Track W session 1 — running notes

Target behaviour: redundant re-fetching (`reread+dup+regrep`) of the Gemma 4 12B coding agent.
Base dev: target_rate 0.733, control_flagged 0.167, control_tool_changed 0.233, no_tool_call 0.008.
(selftest "tiny" random edit, pre-session: 0.700 / 0.117 / 0.167 / 0.008 -> gives a feel for noise.)
Budget: 50 evaluate() calls. Noise threshold ~0.10 (n = 60).

## Log

### Data exploration (no evals)
- Dev split: 622 points (614 with activations), 24 tasks, ALL from Textualize/rich. 123 target-family
  points with activations, 444 unflagged.
- Eval set = first 60 target-family points (record order, i <= 440) + 60 shuffled unflagged.
  Targets: 45 read_file / 15 run_command; flags 28 reread, 14 dup+reread, 11 dup+regrep, 4 regrep, 3 dup.
  Controls: 37 run_command, 18 read_file, 3 edit, 1 write, 1 submit.
- KEY: 43/60 targets are after context compaction (vs 29/60 controls). Re-classifying each target
  against only the calls VISIBLE in its prompt: 32/60 are not redundant w.r.t. the visible context
  (the earlier fetch was summarised away by compaction -> re-reading is rational for the model),
  28/60 are redundant against visible context (in-context loops: e.g. 6x the same
  `grep -n "render_renderables" rich/console.py`, reasoning "Wait, I already did that.").
  Base re-run repeats the compaction-induced ones ~27/32 (0.84) and the visible ones ~16/28 (0.57).
  => even a perfect fix of in-context redundancy alone would only reach ~0.45 target_rate.
- Null-ish edit (selftest "tiny") vs base: 6/60 target and 6/60 control points flip -> per-point
  sampling noise ~10% (seeded, but any logit change re-rolls long thinking traces).
- Architecture (transformers gemma4_unified): h += PostAttnNorm(attn(InNorm h)); h += PostFFNorm(mlp(PreFFNorm h));
  h *= layer_scalar. RMSNorm multiplies by weight (not 1+w). So a down_proj delta reaches the
  residual as ~ gamma_postff * dy / rms(y): MLP writes have bounded norm; edits rotate them.
  Full-attention layers 5,11,...,47 (k_eq_v, no v_proj). Tied embeddings, logit softcap 30.
- Leave-one-task-out diff-in-means AUC at last prompt token (ana1): target vs unflagged peaks
  ~0.67 (L41-46); within read_file ~0.75 (L27-29); within compaction ~0.63; next tool read vs cmd
  ~0.76; edit vs other ~0.79; after_compaction itself ~0.85 (L23-24). Signals are moderate.
- Stratified (tool x compaction) diff-in-means direction r_L (ana2): separates redundant vs fresh READS
  (LOTO AUC 0.65-0.76 at L20-36), not greps (AUC <= 0.5). Logit lens: +r (redundant side) at L40-46 =
  "Wait / Yes / So / Anyway / actually / indeed"; -r = "now / let / NEXT / failed / worked / try";
  at L28-36 -r = "try / attempt / test / should / success / failure" (many languages). I.e. redundant
  reads happen in a deliberating "Wait"-mode, fresh actions in an "act / try" mode.
  Directions rotate across layers (adjacent-layer cos 0.5-0.65).
- On the 120 eval points (ana4): projection predicts controls drifting into redundancy on base re-run
  (AUC 0.64-0.70) but is INVERTED for targets (AUC 0.35-0.45): high-"Wait" targets are the visible loops,
  which the base repeats less reliably; the confident compaction re-reads sit lower.
- Who writes the gap (ana3/5): along r40 mostly MLPs 29-40 (L33 +2.9, L34 +1.8, L38-40 +1.3/+1.7/+2.8,
  ~5.4 of the 6.8 gap at L40 after layer-scalar decay); along r28 MLPs 26-28. Per-neuron: gap spread
  over thousands of neurons; top contributors are readable discourse neurons (L33 n9668 and L34 n13427
  write "wait", L40 n13761 "wait/holy/wow/unexpected" vs "whatever/meh", L35 n13394 "okay", L38 n12025
  "actually", L33 n7916 "yes").
- Edit-vs-explore direction (ana7): ~orthogonal to r (cos 0.03-0.16), AUC ~0.86; logit lens
  +edit = "now / OK / perfect / correct / confirmed", -edit = "search / explore / where / let".
  Targets sit between explore and edit on it.

### E01 `e01_r40abl_mlp29-40` (eval 1)
Edit: remove the r40 component from MLP writes at L29-40 (12 down_proj, rank-1 each, q = norm(gamma_postff*r40)).
At the decision token this removes ~5.4/6.8 of the target-unflagged gap and centres both groups near 0.
Expectation: 0.60-0.65 if "Wait-mode" mediates the loops; ~0.70-0.75 if not causal.
Result: target 0.700 / ctl_flagged 0.200 / ctl_changed 0.233 / no_tool 0.000 -> NULL (all within noise).
Per point: vis targets 15 kept, 2 fixed, 1 new; comp 24 kept, 3 fixed, 2 new; controls churn more than
the null edit (16 flag flips vs 5) -> the edit is active but non-directional.
Conclusion: the decision-token "Wait-mode" direction is a correlate, not a lever (at least via L29-40 MLP writes).

### Template insight (after E01)
- The chat template re-renders every assistant turn's reasoning since the last user message
  (`<|channel>thought ... <channel|><|tool_call>call:NAME{...}<tool_call|><|tool_response>...`), so the
  model sees its own earlier plans ("Let me search for X again") -> loops are self-copying.
- The last prompt token = the `\n` after `<|channel>thought`, i.e. the FIRST position of the new
  thought. The decision-token activations therefore predict the first word of the thought (hence the
  "Wait/Yes/So" logit lens), which explains why r40 was not a lever.
- Tool results carry budget warnings ("Only 0 tool call(s) remaining (28/28 used). Finalize your edits
  and call submit_patch") which the model ignores (i=252 re-reads after 28/28).
- Some "rereads" are boundary overlaps (read 1299-1350 then 1350-1450 is flagged).
- No constant (bias-like) channel: the huge residual dim 1750 (~-200 at L24) is a massive activation of
  the `\n` delimiter and is ~erased by the pre-norm gains; cannot assume it at other positions.
- Along the edit-vs-explore direction e40, L40's MLP writes -4.7 for explore states (89% negative) vs
  +0.3 for edit states at the thought start -> reflecting it is a conditional push toward "commit".

### E02 `e02_e40refl` (eval 2) — causal test of the commit/plan direction (not a candidate)
Edit: reflect (lam=2, norm-preserving) the e40 component of MLP writes L35,39,40 and attn o_proj L37-40.
Estimated push at L40 for explore states ~ +14 (1.5 sd, to edit level); edit states ~ +1.
Expectation if causal: many explore decisions -> edit_file; target_rate well down AND control_tool_changed
well up. If not causal: null like E01.
Result: target 0.700 / ctl_flagged 0.200 / ctl_changed 0.250 / no_tool 0.017 -> NULL. Edit-tool use unchanged
(targets 0 edits; controls 4 edit/2 write/1 submit vs base 3/1/1). Churn similar to E01.
Conclusion: thought-start "plan" directions (redundancy, commit) are not levers even with 1.5-sd pushes.
The action is decided later in the thought; the decision-token state only sets odds (the base re-run
already flips 16/60 targets from the SAME state). Need levers whose meaning is position-independent.
Side probes (ana10, LDA-64PC LOTO): "budget warning in last tool result" AUC 0.94 (L16); late&no-edit 0.81
(L24); visible-loop state ~0.74 (L24-36); target vs unflagged ~0.70. Budget warning present in the last
tool result for 23/60 targets vs 13/60 controls.

### E03 `e03_readtok_abl` (eval 3) — calibration: can a final-layer token edit move tool choice?
Edit: ablate the readout direction of token `read` (id 1399, the first token of `read_file` after
`call:`; the no-space variant, so prose " read" is untouched) = E_read*gamma_final, from MLP down_proj
and attn o_proj writes at L44-47 (8 matrices, lam=1).
Expectation: fewer read_file calls (targets 35/60 base -> ~20), target_rate ~0.55-0.60 (reads ->
grep/run, some become regreps); controls: fewer run->read drifts, so ctl_changed ~flat, ctl_flagged down.
Not selective by construction; it tests whether token-level levers work at all.
Result: target 0.683 / ctl_flagged 0.483 / ctl_changed 0.833 / no_tool 0.850 (!) / 1508 s -> BROKEN.
102/120 outputs had no tool call; the few calls contain junk `<image|>` tokens inside arguments.
Diagnosis: with tied embeddings every token row shares a large common component; the raw readout
direction E_read*gamma_final overlaps what the last layers use to shape ALL logits (incl. suppressing
special tokens), so removing it from 8 late writes scrambles decoding. Lesson: token-level edits must use
centred / contrastive token directions, and should be sanity-checked offline first.
Offline check added: for late-layer edits, recompute the first-thought-token distribution at the 614
decision tokens from the provided L47 residual + recomputed L47 MLP/attn writes (direct path + final
norm + tied unembedding; a logit-lens readout, no forward pass) and report KL / top-token changes.
On E03's deltas the offline screen predicts KL 0.098, 76/614 top-1 changes and `<image|>` becoming top-1
-> it would have caught the breakage. On E01/E02 it shows they were WEAK where it matters: direct-path
KL 0.004/0.007, 18/27 of 614 first tokens changed.

### Lexical-lever screens (offline only, no evals)
- gamma_final is huge and uneven (max 604, mean 20): readout rows share a common component (mean cos 0.31,
  PC1 = 10.7% energy) -> raw token directions are unsafe; need centred/contrastive directions.
- "NoWait" via projection ablation: the Wait logit at the thought start is carried by late ATTENTION
  writes (MLP47-only ablation: P(Wait first) 0.117 -> 0.114). Centred dir, MLP+attn 44-47: P(Wait)
  -> 0.027 but turns 37 "I" openers into "Okay" (KL 0.37). Orthogonalised vs top-50 openers, MLP+attn
  40-47: P(Wait) -> 0.052, KL 0.084. Orthogonalised vs top-200: P(Wait) RISES to 0.158 (the late writes
  along the cleaned direction are suppressive here and ablation removes the suppression). Projections can
  only remove/flip existing components, not add a bias -> lexical suppression this way is unreliable.
- Thoughts opening with "Wait" are redundant 37% (visible-redundant 26%) vs 5% "The", 7% "Now",
  11% "Ah", 24% "Okay" (correlational).

### E04 `e04_actadd_e40` (eval 4) — dose test: strong ActAdd of the commit direction
Construction: rank-1 down_proj deltas at L34-40 that READ the consistent decision-token intermediate
pattern c_L (mean activation direction; cos-to-mean ~0.7-0.8 at decision tokens) and WRITE along
gamma_postff*e40 (the efficient pre-norm direction; a first attempt with e/gamma collapsed the write
norm 28.8 -> 10.2). Each module's write rotates only slightly (cos 0.986-0.992). Direct-path shift at L40
= +23.7 (2.5 sd, 1.7x the explore->edit gap) at decision tokens; first-token KL 0.016, 42/614 openers
change (mostly -> "Okay"). Elsewhere the push scales with c_L^T a at that position (unknown; likely weaker).
Prediction if decision-state steering is a lever at this dose: edit_file share up, target_rate <= 0.60,
control_tool_changed >= 0.33. If null: the thought re-derives the action from context -> stop steering
the decision state.
Result: target 0.683 / ctl_flagged 0.150 / ctl_changed 0.233 / no_tool 0.008 -> NULL for targets (0 edits;
34 read / 24 run / 1 submit / 1 text). Controls: edit 5 (base 3), submit 3 (base 1), write 1 -> at most a
faint commit tilt on controls, within noise. 2.5 sd at the decision state does not move the action.

### E05 `e05_actadd_e40_x4` (eval 5) — same construction at ~4x dose (decisive dose test)
MLP L32-40 down_proj, per-module cos(w1,w0) >= 0.85; direct-path shift at L40 = +100 (10.8 sd, 7x the
explore->edit gap); first-token KL 0.26, 188/614 openers change (mostly -> "Okay"), no special-token leak.
Prediction: if the decision state can steer actions at all, edits/submits jump and target_rate drops
well below 0.6; if still null, decision-state steering is ruled out (the action is re-derived from
context during the thought). Risk: degraded outputs (watch no_tool_call).
Result: target 0.550 / ctl_flagged 0.183 / ctl_changed 0.567 / no_tool 0.292 / 676 s -> DEGRADED, not steered.
Targets: 22 <text> (no tool call), 6 edit + 2 write (vs 0 in base), 18 run, 12 read; garbled args appear
("rich/traceback.,py", "rich/progress. and use the correct path. OK, I will use"). Controls changed 34/60.
The target_rate drop is mostly broken outputs. Some genuine commit appears only at a dose that breaks
the model and is completely non-selective.
CONCLUSION (E01, E02, E04, E05): steering the decision-token state (ablation, reflection, ActAdd at
2.5 sd and 10.8 sd) has no useful window — nothing at moderate doses, degradation + indiscriminate
tool changes at large doses. The action is (re)derived from the context during the thought.

### Programmed suppressor neuron (offline design, supp.py)
One repurposed L47 neuron (lowest-importance n13317): gate = up = beta*d, d = contrastive readout
(target tokens minus mean of 100 common next tokens) mapped into MLP-input space via 1/gamma_pre;
down = -kappa*gamma_post*u. act ~ gelu(z)*z: a one-sided quadratic logit penalty that grows with how
strongly the layer input already predicts the target token (corr(z, P(Wait next)) = 0.83 at decision
tokens). Unlike projection ablation it cannot raise the token anywhere. Sweep for "Wait"/" Wait":
kappa 0.0005: P(Wait first) 0.117 -> 0.083, KL mean 0.017; 0.001: -> 0.056, KL 0.064 (median 0.004);
0.002: -> 0.026, KL 0.22 (median 0.016; changes are Wait -> The/I/It); 0.004: -> 0.009, KL 0.65 (collateral).

### E06 `e06_nowait_supp` (eval 6) — NoWait via one programmed L47 neuron (kappa 0.002)
Hypothesis: the deliberative "Wait" pivot (68% of visible-loop thoughts vs 30% of fresh reads; thoughts
opening with Wait are 37% redundant) drives re-verification fetches. Prediction if true: target_rate
down >= 0.1 with controls ~flat. Risk: "Wait" is also how the model catches itself ("Wait, I already
did that") -> redundancy could go UP. Null = lexical discourse markers do not steer the plan.
Result: target 0.700 / ctl_flagged 0.200 / ctl_changed 0.267 / no_tool 0.050 / 331 s -> NULL.
Per point: visible targets 14 kept, 3 fixed, 2 new; compaction targets 25 kept, 2 fixed, 1 new; controls 8 newly
changed / 6 reverted. no_tool_call rose to 6/120 (3 targets, 3 controls; base 1/120): a slight
degradation signal. The suppressor works as designed offline (P(Wait next) 0.117 -> 0.026 at decision
tokens) but removing the "Wait" pivot does not change WHAT the model decides to fetch: the plan
sentence ("Let me search for X in Y again", "I need to see render_lines") survives without it.

## Conclusion of the session (after 6/50 evaluations) -> stop with an honest null, submit no edit
- None of the 6 edits beats base beyond noise without damage: the "null band" for any mounted LoRA is
  ~0.68-0.70 (pre-session near-null edit 0.700; E01/E02/E04/E06 0.683-0.700); E03 and E05 lower the rate
  only by breaking outputs (no_tool 0.85 / 0.29, ctl_changed 0.83 / 0.57).
- Routes exhausted for a selective edit under these constraints:
  1. decision-token state (the only place activations exist): redundancy and commit directions are
     correlates; ablation / reflection / ActAdd at 2.5 sd do nothing, ActAdd at 10.8 sd degrades the
     model before it steers it, and steers indiscriminately (controls 57% changed).
  2. lexical/discourse level (NoWait programmed neuron): clean offline, null online.
  3. tool-name level (e.g. a read->run suppressor): NOT pursued on purpose. classify() only tracks
     read_file ranges and grep/rg/find patterns, so pushing the tool away from read_file would mostly
     re-route the same re-reads into `sed -n` / `cat` / `head`, which the flagger cannot see -> the metric
     would drop without any call being freed (E05's outputs already show reads turning into shell reads).
  4. attention / retrieval of earlier tool output (the real failure in the visible loops): no activations
     at in-thought or tool-call positions, so any design would be blind.
- Structural ceiling: 32/60 dev targets are re-fetches of content that context compaction removed from
  the prompt (base repeats 27/32); to edit after compaction the agent must re-read the exact lines
  (edit_file needs old_string). Even a perfect fix of in-context redundancy would leave target_rate
  ~0.45; a weight edit that removed those reads would hurt the agent.

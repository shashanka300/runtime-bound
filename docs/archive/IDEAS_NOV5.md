# External research, 2026-11-05 (Kaggle forum + papers): ideas to beat 0.29

LB context: the public split has 58 tasks (1 task = 0.017). Top: Yurnero 0.29; about 9 teams at 0.24; a large plateau at 0.22.
Identical agent files score 0.12-0.22 on the public LB (750974); 15% of tasks flip between two identical
runs (747168). Our lb04 = 0.15 sits at the low end of that noise band.

## Evidence from the forum vs our lb04 (temp 0.2, 28 calls, 7 min, 240 s cmd, think 4096, all 3 graph tools)
| finding | source | lb04 | action |
|---|---|---|---|
| tool budget 50 -> 25 calls: -14/129 tasks (largest effect measured) | 746250 | 28 calls | test 40-50 calls under the time cap |
| temp 0.4/1.0 = 51/129 vs 0.2 = 48/129 (noise-level) | 746250 | 0.2 | test 1.0 |
| graph tools: no benefit; semantic search always empty | 746250 | all 3 on | drop search_similar_code + get_code_subgraph |
| vLLM gemma4 parser loops FOREVER on array args with ']' -> freezes server, all later tasks score 0 | 753973 | get_code_subgraph takes a list | drop array-arg tools (a hidden-run killer) |
| edit_file old_string with docstrings mangled by parser -> 27-51 identical retries | 747168, 756051 | none | prompt/skill: avoid triple quotes in old_string; retry guard |
| read_file truncates at 150 lines -> repeated partial reads | 755291 | none | prompt: always pass start/end lines, grep first |
| thought summaries off: 35 vs 51 (-16) | 746250 | on | keep |
| finalize/patch-hygiene prompt cuts test-patch conflicts 8-11 -> 2-4 | 747857 | ? | keep or add hygiene step |
| stronger teachers only +4-5/124 in same harness; ceiling = harness (32k ctx + compaction) | 747168 | - | supports paper thesis |
| 12B < 31B in every setting (writes 3x more) | 750974 | - | consistent with our 7/39 |
| observation masking matches LLM summarisation at half the cost | arXiv 2508.21433 | ADK summariser | candidate compaction fix |
| verifier-based best-of-N (R4P 26->33%, SWE-Review early exit) | 2510.22775, 2607.06065 | - | no time budget for N>1 at 31B |

## Ranked plan
1. lb07 = lb04 - {search_similar_code, get_code_subgraph} + temp 1.0 + 40 calls + prompt rules (read ranges,
   no triple-quote old_string). Low risk; protects against the parser freeze.
2. Context/compaction fix (keep edited-file content through compaction; repeated-call guard), from Track W.
3. Variance: resubmit the best config once to measure our own noise before trusting a 0.02 difference.

## Addendum (evening): non-obvious levers found in the harness code + papers
- **Skills run real Python inside the sandbox.** swegemma's agent_runner wires SkillToolset to
  AdkSandboxCodeExecutor (load_skill / load_skill_resource / run_skill_script). A bundle can therefore
  ship its own tools as skill scripts. Callbacks cannot be shipped (closed registry), compaction is host-set.
- Risk: run_skill_script takes `args`/`positional_args` lists -> exposed to the gemma4 array-arg parser
  freeze (forum 753973). Mitigation: one bootstrap call (no args) that copies our tools to /tmp/tools;
  afterwards the agent calls them through run_command (a plain string).
- Tools worth shipping (papers):
  1. locate.py: BM25 + exact-identifier grep over the repo from issue text -> top-10 files/lines. Forum
     (hitarthjain0): first grep hit right 30%, grep+BM25 top-5 61%, top-10 74%. SWE-Tester / SweRank /
     SWE-Fixer: coarse BM25 then rerank is the standard cheap localiser.
  2. outline.py: AST outline of a file (class/def with line ranges) so one call reads exactly the
     function (Agentless: hierarchical file -> function -> lines localisation; function level is best,
     arXiv 2604.00167).
  3. replace.py: exact unique replacement reading old/new text from heredoc files, avoiding the
     edit_file docstring/parser mangling (forum 747168, 756051).
- Not feasible under the 12 h budget: best-of-N with a verifier (R4P, SWE-Review); union of 4 runs is
  73/123 vs 54 single (forum 747168), but without hidden tests there is no reliable selector.

## Addendum 2: lessons from newer non-Gemma models / 2026 harness studies
- Harness-design study (arXiv 2609.20804; Nemotron-3 30B is the closest analogue to Gemma 31B, 32k window):
  * context management at 32k: none 9.4% -> summarisation 23.8% SWE-bench Verified (ADK compaction is a T3 analogue).
  * PLANNING TOOL (update_plan todo list, re-shown every turn): 13.6% -> 25.2% (+11.6) for the 30B;
    runs ending with no edit 68.6% -> 27.8%. Barely matters for 550B models. => biggest transferable lever.
  * predefined file/search/edit tools vs bash-only: 25.2% vs 10.2% for the 30B (weak models need
    purpose-built tools; strong ones do not) => supports lb09's helper tools.
  * "recall" of elided content: no gain. Their summary schema: Goal / Files touched / Done / Pending /
    Errors & fixes / Current state / Next step.
- Qwen3-Coder-Next report (2603.00729): JSON tool calls with multi-line code are escaping-fragile (they built
  an XML format for string-heavy args); scaffold choice swings other models by 0-40 points; rule-based
  tool-call validation cuts invalid calls/retries. => route multi-line edits through a heredoc (replace.py).
- Ornith-1.0-9B (built on Gemma 4 / Qwen 3.5 bases; RL on rollouts + scaffold): 69.4 vs Gemma4-31B 52 on
  Verified in OpenHands at 256k context, temp 1.0 / top_p 0.95 (recommended 0.6 / top_k 20). Not usable
  here (one fixed base model), but confirms training is where the large gains are.
- Next candidate (lb10): planning emulation. A structured /tmp/plan.md (Goal/Files/Done/Pending/Next) kept
  by the agent, combined with lb09's tools; if lb08's notes help, merge them into this schema.

## Addendum 3 (2026-11-06): thinking repetition loops are the biggest time sink

Trace profile of the 39-task screens (tool execution is only ~10 s/task; generation is the time):
- lb04: 32 steps (in 21/39 tasks) produce >1800 completion tokens; 45 of the 60 steps >=1k tokens repeat
  one line >=5 times (e.g. "Wait, `?` is 63." x50). These steps run to the 4096 thinking budget, take a
  median 58 s each (local; task cap 262 s) and use 42% of all completion tokens / 28% of generation time.
- Tasks with a loop resolve 4/21; tasks without resolve 12/18. Same pattern in lb05b/07/08/09/10.
- temperature 1.0 (lb07) does not reduce loops (31 loop steps). thinking_budget 2048 (lb05a) halves the
  time lost (1093 s vs 1887 s) but loops still occur (29 steps).
- Known Gemma 4 model-level bug (google-deepmind/gemma#622, vllm#40080); penalties reported unreliable;
  serving-side fixes (attention backend) are not under bundle control.
- Bundle levers (all allowed by GenerationConstraints): thinking_budget, max_output_tokens (hard cap per
  step), presence/frequency_penalty, prompt rule to keep thinking short.
- Other 2026 work read: harness study 2609.20804 (planning + stuck reminder after 5 identical calls),
  Same Model Different Harness 2608.26218 (age-tiered tool-output shortening + stall detector at 20k
  window, 43 -> 72 solves; harness-side), FailFast/RestartSmart 2608.03222 (restart, harness-side),
  SWE-Edit 2604.26102 (editor subagent), FL studies 2609.00854 / 2610.25826 (localization adds little).

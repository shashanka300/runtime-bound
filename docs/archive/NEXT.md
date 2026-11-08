# Next steps (updated 2026-09-28 ~10:00 IST)

## Where we are
- CORRECTED 2026-09-28 afternoon: the host says the hidden tasks come from DIFFERENT repos and
  gold patches validate at 100% on the scorer; the public-set gradability problem (41/129,
  being re-measured cleanly in data/probes/official_gold_v2.jsonl) does not explain the board.
  The remaining gap = unseen repositories + frontier-filtered difficulty. See design_changes.md.
- (Superseded) the public offline grading environment can grade only 41/129 public tasks
  (rich 39/48, httpx 1/1, fastapi 1/67, requests 0/13).
- Evidence notebook + draft forum post (not posted; the user decides):
  reports/gradability_audit/gradability_audit.ipynb.
- Best configuration so far: thinking off, max_output 4096, no LoRA (lb03, E1b arm G).
- Progress tracker: tools/progress_log.sh -> data/logs/progress.log (one line / 10 min).

## Forum status (2026-09-28 ~10:30 IST)
- The gradability report is posted by the user (thread 743973, with the executed notebook); no
  host reply yet. Check it first.
- Host emphasis (Elan Markowitz's two welcome posts): post-training (PEFT/RL), graph-based code
  reasoning with the provided graphs/embeddings, consumer-grade hardware; Paper Track awards are
  Best Paper, Best New Resource, Best New Application. Up to two final submissions per team
  (whether one team may enter every award is unanswered: paper-track thread 743255).
- Watch 743964 (ParallelAgent concurrency on the scorer, LoRA ETA, self-generated training data).

## Plan agreed in principle (user to pick the order)
- Two writeups: Best Paper = the transfer-gap decomposition on consumer hardware; Best New
  Resource = the evaluation suite (swebox, fidelity mode, stall audit, gradability audit), no
  competition data redistributed.
- E4: code-graph/embedding localization tools, evaluated in fidelity mode on gradable tasks (ties
  the paper to the host's graph dataset).

## Proposed next work (to confirm with the user)
1. Re-base local evaluation: a `gradable41` subset (the officially gradable public tasks) and/or
   run swebox with the official offline dependency set, so every local number predicts the board.
   Small n (41; 15 in dev40) -> use more seeds.
2. Agent work aimed at gradable task types (mostly rich): prompt/efficiency (fewer turns, shorter
   outputs), ADK skills (localisation / targeted-test scripts; needs swebox skill support).
3. Fine-tuning: prepare RFT data export + QLoRA pipeline; train only after the host fixes LoRA
   loading in the scorer (forum 743508); confirm with a canary adapter submission.
4. Paper: start the outline around the transfer-gap decomposition (ladder, replay mechanism,
   stall correction, gradability cap) plus the loop/context results.

## Restart services after a reboot
```bash
cd ~/challeenge
bash infra/services.sh
setsid -f bash infra/vllm/serve_submission.sh > data/logs/serve_restore.out 2>&1 < /dev/null
setsid -f bash tools/progress_log.sh 600 < /dev/null > /dev/null 2>&1
```

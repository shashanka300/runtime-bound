"""Build LoRA SFT examples from official-harness traces (docs/design_changes.md, 2026-11-06).

Keeps runs that swebox graded as resolved (<run>/regrade.jsonl, or results.jsonl if absent), then:
  - restores the hint-free task prompt (drops the harness's "## Hints:" section);
  - cleans thinking loops: repeated lines inside one thought are dropped after their first
    occurrence, and a thought is capped at --max-thought-chars;
  - in hinted runs, drops the sentences that cite the hint; rejects runs that edit before any
    search/read, or that still mention it;
  - caps the number of runs per task (--per-task) so easy tasks do not dominate.
Output: one JSON object per line, {"id", "source", "messages": [...], "stats": {...}} with
messages in a neutral chat form (system / user / assistant{thinking, tool_calls} / tool).

    python3 tools/build_sft.py data/lora/gen_plain data/lora/gen_hint1 --out data/lora/sft_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

EDIT_TOOLS = {"edit_file", "write_file", "apply_patch", "str_replace", "replace_in_file"}
LOOK_TOOLS = {"run_command", "read_file", "search_code", "grep", "list_files", "find_files"}
HINT_WORDS = re.compile(r"(?<!type )\b(hint|hints|maintainer|reference change|reference fix)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def redact_hints(text: str) -> tuple[str, int]:
    """Drop the sentences of a hinted run that cite the hint (the model often writes 'The hint says
    the fix is in X.'); the rest of the reasoning and every tool call stay."""
    n = 0
    lines = []
    for line in text.split("\n"):
        sents = SENT.split(line)
        keep = [s for s in sents if not HINT_WORDS.search(s)]
        n += len(sents) - len(keep)
        if keep or not line.strip():
            lines.append(" ".join(keep))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), n


def strip_hints(prompt: str) -> str:
    return re.sub(r"## Hints:\n.*?(?=\n## |\Z)", "", prompt, count=1, flags=re.S).replace("\n\n\n", "\n\n")


def clean_thought(text: str, max_chars: int) -> tuple[str, int]:
    seen: set[str] = set()
    kept, dropped = [], 0
    for line in text.splitlines():
        key = re.sub(r"\s+", " ", line.strip().lower())
        if len(key) >= 20 and key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(line)
    out = "\n".join(kept).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0].strip()
        dropped += 1
    return out, dropped


def resolved_ids(run: Path) -> set[str]:
    f = run / "regrade.jsonl"
    if not f.exists():
        f = run / "results.jsonl"
    return {json.loads(ln)["instance_id"] for ln in f.open() if ln.strip() and json.loads(ln)["resolved"]}


def convert(trace: dict, hinted: bool, max_chars: int) -> tuple[list[dict], dict] | None:
    msgs: list[dict] = []
    stats = collections.Counter()
    first_edit = first_look = None
    n_call = 0
    for s in trace["steps"]:
        ev = s["extra"].get("event_type")
        if s["source"] == "system":
            msgs.append({"role": "system", "content": s["message"]})
        elif s["source"] == "user":
            msgs.append({"role": "user", "content": strip_hints(s["message"]) if hinted else s["message"]})
        elif s["source"] == "agent":
            if s["extra"].get("author") not in (None, "swe_agent"):
                return None  # sub-agent traces are not used
            text = s.get("message") or ""
            if hinted:
                text, n = redact_hints(text)
                stats["hint_sentences_dropped"] += n
                if HINT_WORDS.search(text):
                    stats["hint_mention"] += 1
            calls = s.get("tool_calls") or []
            if ev == "final" and not calls:
                msgs.append({"role": "assistant", "content": text})
                continue
            thought, dropped = clean_thought(text, max_chars)
            stats["loop_lines_dropped"] += dropped
            stats["thought_chars"] += len(thought)
            msgs.append({"role": "assistant", "thinking": thought,
                         "tool_calls": [{"id": c["tool_call_id"], "name": c["function_name"],
                                         "arguments": c["arguments"]} for c in calls]})
            obs = (s.get("observation") or {}).get("content")
            for c in calls:
                n_call += 1
                if c["function_name"] in EDIT_TOOLS and first_edit is None:
                    first_edit = n_call
                if c["function_name"] in LOOK_TOOLS and first_look is None:
                    first_look = n_call
                msgs.append({"role": "tool", "tool_call_id": c["tool_call_id"], "name": c["function_name"],
                             "content": obs if isinstance(obs, str) else json.dumps(obs)})
    stats["calls"] = n_call
    stats["first_edit"] = first_edit or 0
    if hinted and (stats["hint_mention"] or first_edit is None or first_look is None or first_look > first_edit):
        return None
    return msgs, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--max-thought-chars", type=int, default=6000)
    args = ap.parse_args()
    per = collections.Counter()
    kept = rejected = 0
    with args.out.open("w") as fh:
        for run in args.runs:
            hinted = "hint" in run.name
            for iid in sorted(resolved_ids(run)):
                if per[iid] >= args.per_task:
                    continue
                p = run / "harness" / iid / "traces" / f"trace_{iid}.json"
                if not p.exists():
                    continue
                res = convert(json.loads(p.read_text()), hinted, args.max_thought_chars)
                if res is None:
                    rejected += 1
                    continue
                msgs, stats = res
                per[iid] += 1
                kept += 1
                fh.write(json.dumps({"id": f"{run.name}/{iid}", "source": run.name, "hinted": hinted,
                                     "messages": msgs, "stats": stats}) + "\n")
    print(f"kept {kept} runs from {len(per)} tasks; rejected {rejected}")


if __name__ == "__main__":
    main()

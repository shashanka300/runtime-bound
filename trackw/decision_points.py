"""Track W2: label every recorded agent request (a decision point) with the action the model
took and whether that action was budget-wasting, using the same rules as tools/behaviour_scan.py
but on the recorded request/response stream (so labels line up with exact vLLM token ids).

    python3 trackw/decision_points.py data/screen_nov/lb06_12b_requests.jsonl
writes <records>.points.jsonl: {"i", "task", "k" (call index in task), "tool", "args",
"flags", "first_edit_before"}; prints per-flag counts.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def task_key(body: dict) -> str:
    user = next(m for m in body["messages"] if m["role"] == "user")
    content = user["content"]
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    m = re.search(r"repository (\S+)\.\s+Problem Statement:\s*(.{0,120})", content, re.S)
    return (m.group(1) + "|" + m.group(2)) if m else content[:160]


LOOP = "<|channel>thought\n<channel|><|channel>thought"


def channel_loop(resp: dict) -> bool:
    """Degenerate 12B output: empty thought channel opened and closed repeatedly to the limit."""
    msg = resp["choices"][0]["message"]
    return LOOP in ((msg.get("content") or "") + (msg.get("reasoning") or ""))


def action(resp: dict) -> tuple[str, dict]:
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return "<text>", {}
    f = calls[0]["function"]
    try:
        args = json.loads(f.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {"_raw": f.get("arguments")}
    return f["name"], args


def new_state() -> dict:
    return {"seen": [], "reads": {}, "greps": [], "k": 0, "edited": False}


def _span(args: dict) -> tuple[int, int]:
    try:
        s = int(args.get("start_line") or 1)
        e = int(args.get("end_line") or s + 149)
    except (TypeError, ValueError):
        s, e = 1, 150
    return s, e


def _grep(args: dict) -> str | None:
    m = re.search(r"\b(grep|rg|find)\b(.*)", str(args.get("command", "")))
    return re.sub(r"\s+", " ", m.group(2)).strip() if m else None


def classify(st: dict, name: str, args: dict) -> list[str]:
    """Budget-wasting flags for taking (name, args) given the task history `st` (not mutated)."""
    flags = []
    if json.dumps([name, args], sort_keys=True) in st["seen"]:
        flags.append("dup")
    path_ = args.get("filepath") or args.get("path") or ""
    if name == "read_file" and path_:
        s, e = _span(args)
        if any(a <= e and s <= b for a, b in st["reads"].get(path_, [])):
            flags.append("reread")
    if name == "run_command" and _grep(args) in st["greps"]:
        flags.append("regrep")
    if any(k.strip('"') != k for k in args):
        flags.append("badkey")
    return flags


def update(st: dict, name: str, args: dict) -> None:
    st["seen"].append(json.dumps([name, args], sort_keys=True))
    path_ = args.get("filepath") or args.get("path") or ""
    if name == "read_file" and path_:
        st["reads"].setdefault(path_, []).append(list(_span(args)))
    if name == "run_command" and _grep(args) is not None:
        st["greps"].append(_grep(args))
    st["k"] += 1
    if name in ("edit_file", "write_file"):
        # after an edit, re-running commands and re-reading the edited file is verification
        st["edited"] = True
        st["seen"] = [x for x in st["seen"] if not x.startswith('["run_command"')]
        st["reads"].pop(path_, None)


def main() -> None:
    path = Path(sys.argv[1])
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    state: dict[str, dict] = defaultdict(new_state)
    out = []
    summaries: list[dict] = []
    task = None
    for i, r in enumerate(recs):
        resp = r.get("response")
        if not isinstance(resp, dict) or "choices" not in resp:
            continue
        msgs = r["request"]["messages"]
        first_user = task_key(r["request"])
        if len(msgs) == 1 and first_user.startswith("The following is a conversation history"):
            # ADK events-compaction summariser call, not an agent decision; count its loops
            summaries.append(
                {
                    "i": i,
                    "task": task,
                    "loop": channel_loop(resp),
                    "tokens": resp.get("usage", {}).get("completion_tokens"),
                    "latency_s": r.get("latency_s"),
                }
            )
            continue
        if "|" in first_user and len(msgs) <= 2:
            task = first_user  # tasks run one at a time: a fresh task prompt starts a task
        st = state[task]
        name, args = action(resp)
        loop = channel_loop(resp)
        out.append(
            {
                "i": i,
                "task": task,
                "k": st["k"],
                "tool": name,
                "args": json.dumps(args)[:200],
                "flags": classify(st, name, args) + (["chanloop"] if loop else []),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
                "latency_s": r.get("latency_s"),
                "first_edit_before": st["edited"],
                "after_compaction": first_user.startswith("For context:"),
                "state": json.loads(json.dumps(st)),
            }
        )
        update(st, name, args)
    dst = path.with_suffix(".points.jsonl")
    dst.write_text("".join(json.dumps(p) + "\n" for p in out))
    c = Counter(f for p in out for f in p["flags"])
    print(f"{len(out)} decision points in {len(state)} tasks; flags {dict(c)}")
    print(f"tools {dict(Counter(p['tool'] for p in out))}")
    loop_tasks = {p["task"] for p in out if "chanloop" in p["flags"]}
    loop_tasks |= {s["task"] for s in summaries if s["loop"]}
    lost = sum(p["latency_s"] or 0 for p in out if "chanloop" in p["flags"])
    lost += sum(s["latency_s"] or 0 for s in summaries if s["loop"])
    print(
        f"channel loop: {c.get('chanloop', 0)} agent steps, "
        f"{sum(s['loop'] for s in summaries)}/{len(summaries)} summariser calls, "
        f"in {len(loop_tasks)}/{len(state)} tasks, {lost:.0f}s of generation lost"
    )
    path.with_suffix(".summaries.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in summaries)
    )
    print(dst)


if __name__ == "__main__":
    main()

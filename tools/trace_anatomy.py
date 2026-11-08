"""Where does a run's wall-clock go? Per-task turn anatomy of official-harness traces.

For each task in a screen directory (tools/official_screen.py output), reads the ATIF trace and
reports: agent turns, output tokens, the longest turn, turns of >= --long tokens, the share of
time spent in the 3 slowest turns, time to the first edit, and repetition loops (a turn of
>= --loop-tokens tokens in which more than 25% of the sentences are duplicates). Summaries are
grouped by outcome (resolved / timed out / failed otherwise).

    python3 tools/trace_anatomy.py data/screen_nov/lb04_public_7min [more screen dirs]
    python3 tools/trace_anatomy.py RUN --ids a,b,c      # restrict to a task subset
    python3 tools/trace_anatomy.py RUN --json out.json  # per-task rows
"""

from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import re
import statistics as st
from pathlib import Path


def _d(x):
    return x if isinstance(x, dict) else ast.literal_eval(x) if x else {}


def _l(x):
    return x if isinstance(x, list) else ast.literal_eval(x) if x else []


def loop_share(msg: str) -> float:
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", msg or "") if len(x.strip()) > 20]
    if not s:
        return 0.0
    c = collections.Counter(s)
    return sum(v - 1 for v in c.values() if v > 1) / len(s)


def task_rows(run: Path, ids: set[str] | None, long_tok: int, loop_tok: int) -> list[dict]:
    res = {json.loads(l)["instance_id"]: json.loads(l) for l in open(run / "results.jsonl")}
    rows = []
    for iid, r in res.items():
        if ids and iid not in ids:
            continue
        fs = glob.glob(str(run / "harness" / iid / "traces" / "*.json"))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        t0 = prev = float(_d(d["steps"][0]["extra"])["elapsed_s"])
        gaps, toks, first_edit, loops = [], [], None, 0
        for s in d["steps"]:
            if s.get("source") != "agent":
                continue
            t = float(_d(s["extra"])["elapsed_s"])
            gaps.append(t - prev)
            prev = t
            c = _d(s.get("metrics")).get("completion_tokens", 0)
            toks.append(c)
            if c >= loop_tok and loop_share(s.get("message")) > 0.25:
                loops += 1
            if first_edit is None and any(
                    x.get("function_name") in ("edit_file", "write_file") for x in _l(s.get("tool_calls"))):
                first_edit = t - t0
        err = r.get("error") or ""
        outcome = "resolved" if r["resolved"] else "timed_out" if "timeout" in err else "failed"
        top3 = sum(sorted(gaps, reverse=True)[:3])
        rows.append(dict(id=iid, outcome=outcome, turns=len(toks), out_tokens=sum(toks),
                         longest_turn=max(toks or [0]), long_turns=sum(c >= long_tok for c in toks),
                         top3_share=top3 / max(sum(gaps), 1e-9), first_edit_s=first_edit,
                         loops=loops, duration_s=r.get("duration_s"), patch_chars=r.get("patch_chars")))
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fe = [r["first_edit_s"] for r in rows if r["first_edit_s"] is not None]
    print(f"  {name:10} n={len(rows):3}  turns {st.mean(r['turns'] for r in rows):5.1f}  "
          f"out-tok {st.mean(r['out_tokens'] for r in rows):6.0f}  "
          f"longest (median) {st.median(r['longest_turn'] for r in rows):5.0f}  "
          f"long turns {st.mean(r['long_turns'] for r in rows):3.1f}  "
          f"top-3 share {st.mean(r['top3_share'] for r in rows):4.0%}  "
          f"first edit (median) {st.median(fe) if fe else float('nan'):4.0f}s  "
          f"no edit {sum(r['first_edit_s'] is None for r in rows)}  loops {sum(r['loops'] for r in rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ids", default="")
    ap.add_argument("--long", type=int, default=1500, help="a 'long' turn, in output tokens")
    ap.add_argument("--loop-tokens", type=int, default=3000, help="minimum turn length for a loop")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    ids = set(a.ids.split(",")) if a.ids else None
    out = {}
    for run in a.runs:
        rows = task_rows(run, ids, a.long, a.loop_tokens)
        out[str(run)] = rows
        print(run)
        summarize("all", rows)
        for o in ("resolved", "timed_out", "failed"):
            summarize(o, [r for r in rows if r["outcome"] == o])
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

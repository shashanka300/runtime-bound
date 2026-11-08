"""Track W1: classify every tool call in official-harness traces to find budget-wasting
behaviours (docs/archive/PLAN_NOV.md, Track W).

    python3 tools/behaviour_scan.py data/screen_nov/lb04_public_7min [more screen dirs]

Per call: tool, arguments, and flags
  dup      identical (tool, arguments) to an earlier call in the same task
  reread   read_file whose line range overlaps a range of the same file already read
  regrep   run_command grep/rg/find whose pattern was already searched
  err      the observation reports an error (status != ok, non-zero exit, "Error")
  retry    the same tool on the same file right after an err (e.g. failed edit_file)
Per task: calls, wasted calls (any flag but err), first edit step, submitted, resolved.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def _range(args: dict) -> tuple[int, int]:
    s = args.get("start_line") or 1
    e = args.get("end_line") or (int(s) + 149)
    return int(s), int(e)


def _is_err(obs: str) -> bool:
    return bool(
        re.search(r'"status\\?":\s*\\?"(error|fail)', obs)
        or re.search(r'exit_code\\?":\s*[1-9]', obs)
        or "Traceback" in obs
        or re.search(r"\bError\b", obs[:400])
    )


def scan_task(trace: Path) -> dict:
    d = json.loads(trace.read_text())
    seen: set[str] = set()
    reads: dict[str, list[tuple[int, int]]] = {}
    greps: set[str] = set()
    calls = []
    prev_err_key = None
    for step in d["steps"]:
        for tc in step.get("tool_calls") or []:
            name, args = tc["function_name"], tc.get("arguments") or {}
            key = json.dumps([name, args], sort_keys=True)
            obs = json.dumps(step.get("observation") or "")
            flags = []
            if key in seen:
                flags.append("dup")
            seen.add(key)
            path = args.get("filepath") or args.get("path") or ""
            if name == "read_file" and path:
                r = _range(args)
                if any(a <= r[1] and r[0] <= b for a, b in reads.get(path, [])):
                    flags.append("reread")
                reads.setdefault(path, []).append(r)
            if name == "run_command":
                m = re.search(r"\b(grep|rg|find)\b(.*)", args.get("command", ""))
                if m:
                    pat = re.sub(r"\s+", " ", m.group(2)).strip()
                    if pat in greps:
                        flags.append("regrep")
                    greps.add(pat)
            if prev_err_key and prev_err_key == (name, path) and path:
                flags.append("retry")
            err = _is_err(obs)
            if err:
                flags.append("err")
            prev_err_key = (name, path) if err else None
            if name in ("edit_file", "write_file") and not err:
                # after a successful edit, re-running commands and re-reading the edited file
                # is legitimate verification, not waste
                seen = {k for k in seen if not k.startswith('["run_command"')}
                reads.pop(path, None)
            calls.append(
                {
                    "step": step["step_id"],
                    "tool": name,
                    "flags": flags,
                    "args": json.dumps(args)[:160],
                    "obs": obs[:200],
                    "elapsed_s": (tc.get("extra") or {}).get("elapsed_s"),
                }
            )
    tools = [c["tool"] for c in calls]
    first_edit = next((i for i, t in enumerate(tools) if t in ("edit_file", "write_file")), None)
    return {
        "task": trace.parent.parent.name,
        "calls": len(calls),
        "wasted": sum(any(f != "err" for f in c["flags"]) for c in calls),
        "flags": Counter(f for c in calls for f in c["flags"]),
        "tools": Counter(tools),
        "first_edit_call": first_edit,
        "submitted": "submit_patch" in tools,
        "completion_tokens": d["final_metrics"]["total_completion_tokens"],
        "calls_detail": calls,
    }


def main() -> None:
    rows = []
    for screen in map(Path, sys.argv[1:]):
        res = {}
        if (screen / "results.jsonl").exists():
            for ln in (screen / "results.jsonl").read_text().splitlines():
                r = json.loads(ln)
                res[r["instance_id"]] = r
        for trace in sorted(screen.glob("harness/*/traces/trace_*.json")):
            row = scan_task(trace)
            r = res.get(row["task"], {})
            row.update(
                variant=screen.name,
                resolved=r.get("resolved"),
                cap=(
                    "time"
                    if "timeout" in (r.get("error") or "")
                    else "calls"
                    if row["calls"] >= 28
                    else "-"
                ),
            )
            rows.append(row)
    print(
        f"{'variant':<18}{'task':<16}{'ok':<4}{'cap':<6}{'calls':>5}{'waste':>6}{'1st edit':>9}"
        f"{'sub':>4}{'ctok':>7}  flags"
    )
    for r in rows:
        print(
            f"{r['variant'][:17]:<18}{r['task']:<16}{'Y' if r['resolved'] else 'n':<4}{r['cap']:<6}"
            f"{r['calls']:>5}{r['wasted']:>6}{r['first_edit_call']!s:>9}"
            f"{'Y' if r['submitted'] else 'n':>4}{r['completion_tokens']:>7}  {dict(r['flags'])}"
        )
    failed = [r for r in rows if not r["resolved"]]
    tot = Counter()
    for r in failed:
        tot.update(r["flags"])
    print(
        f"\nfailed tasks: {len(failed)}; calls {sum(r['calls'] for r in failed)}; "
        f"wasted {sum(r['wasted'] for r in failed)}; flag totals {dict(tot)}"
    )
    for flag in ("dup", "reread", "regrep", "retry", "err"):
        n = sum(1 for r in failed if r["flags"].get(flag))
        print(f"  {flag:<7} present in {n}/{len(failed)} failed tasks")
    out = Path(sys.argv[1]) / "behaviour_scan.json"
    out.write_text(json.dumps(rows, indent=1, default=dict))
    print(out)


if __name__ == "__main__":
    main()

"""LoRA data synthesis (docs/design_changes.md, 2026-11-06): copies of tasks.jsonl whose hints_text
points at the gold fix, for hint-guided rejection sampling on tasks the agent fails. The hint is
rendered by the harness as a "## Hints" section; training examples are rebuilt with the original,
hint-free problem statement (tools/build_sft.py).

  level 1: the files the reference fix changes (and which ones are new)
  level 2: level 1 + the functions/classes it touches and the size of the change

    python3 tools/make_hinted_tasks.py --out data/lora
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def files(patch: str) -> list[tuple[str, bool]]:
    out = []
    for block in re.split(r"(?m)^--- ", patch)[1:]:
        m = re.search(r"(?m)^\+\+\+ b/(\S+)", block)
        if m:
            out.append((m.group(1), block.startswith("/dev/null") or "@@ -0,0 " in block))
    return out


def symbols(patch: str) -> list[str]:
    seen: list[str] = []
    for m in re.finditer(r"(?m)^@@ [^@]+ @@ ?(.*)$", patch):
        ctx = m.group(1).strip()
        s = re.search(r"(?:def|class)\s+(\w+)", ctx)
        if s and s.group(1) not in seen:
            seen.append(s.group(1))
    for m in re.finditer(r"(?m)^\+\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen[:12]


def hint(patch: str, level: int) -> str:
    fs = files(patch)
    lines = ["A maintainer who fixed this issue changed these files:"]
    lines += [f"- {f}{' (new file)' if new else ''}" for f, new in fs]
    if level >= 2:
        syms = symbols(patch)
        if syms:
            lines.append("The change is in or next to: " + ", ".join(f"`{s}`" for s in syms) + ".")
        n = sum(1 for x in patch.splitlines() if x[:1] in "+-" and x[:3] not in ("+++", "---"))
        lines.append(f"The reference change is about {n} changed lines.")
    lines.append("Use this to go straight to the right place, then read the code and make the fix.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lora")
    ap.add_argument("--tasks", type=Path, default=ROOT / "data/competition/tasks.jsonl")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in args.tasks.open() if ln.strip()]
    for level in (1, 2):
        with (args.out / f"tasks_hint{level}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "hints_text": hint(r["patch"], level)}) + "\n")
    print(rows[0]["instance_id"], "\n" + hint(rows[0]["patch"], 2))


if __name__ == "__main__":
    main()

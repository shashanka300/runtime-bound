"""Grade the patches of an official-harness run (tools/official_screen.py output) with swebox's
own grader, which can grade all 129 public tasks (the official one grades 41; the rest fail on
modules missing from its verification image). Writes <run>/regrade.jsonl, resumable.

    uv run python tools/regrade_swebox.py data/lora/gen_plain [--concurrency 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from swebox import sandbox as sb
from swebox.tasks import load_tasks
from swebox.verify import verify


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    tasks = {t.instance_id: t for t in load_tasks()}
    rows = [json.loads(ln) for ln in (args.run / "results.jsonl").open() if ln.strip()]
    out = args.run / "regrade.jsonl"
    done = {json.loads(ln)["instance_id"] for ln in out.open()} if out.exists() else set()
    dc = sb.client()
    await asyncio.to_thread(sb.ensure_base_image, dc)
    limits = sb.Limits(8.0, 4.0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(row: dict) -> None:
        iid = row["instance_id"]
        p = args.run / "harness" / iid / "patches" / f"{iid}.patch"
        patch = p.read_text() if p.exists() else ""
        async with sem:
            if patch.strip():
                v = await asyncio.to_thread(verify, dc, tasks[iid], patch, limits)
                rec = {"instance_id": iid, "resolved": v.resolved, "applied": v.applied,
                       "exit_code": v.exit_code, "error": v.error, "seconds": v.seconds}
            else:
                rec = {"instance_id": iid, "resolved": False, "applied": False,
                       "exit_code": None, "error": "empty patch", "seconds": 0}
        rec["official_resolved"] = row.get("resolved")
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(iid, rec["resolved"], rec["error"] or "", flush=True)

    await asyncio.gather(*(one(r) for r in rows if r["instance_id"] not in done))


if __name__ == "__main__":
    asyncio.run(main())

"""Grade the reference (gold) patch, and no patch, through the scorer's REAL Phase-2 path:
swegemma.harness.verification.verify_task, called exactly as swegemma.evaluate.Evaluator does
(snapshot paths from resolve_task_snapshot_paths, competition wheels, docker or subprocess
sandbox). Decides whether the offline grading environment can grade a task at all.

    infra/official/.venv/bin/python tools/official_verify_gold.py --ids a,b [--sandbox docker]
Writes data/probes/official_verify_<sandbox>.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "data" / "competition"


async def grade(ev: Evaluator, task, patch: str) -> dict:  # type: ignore[no-untyped-def]
    snap, base, patch_path = resolve_task_snapshot_paths(
        COMP / "snapshots", task.instance_id, task.repo
    )
    t0 = time.perf_counter()
    res = await verify_task(
        ev.docker,
        ev.config,
        task,
        snap,
        base_snapshot_path=base,
        patch_path=patch_path,
        agent_patch=patch,
        start_time=t0,
    )
    return {
        "resolved": res.resolved,
        "exit": res.test_exit_code,
        "error": res.error,
        "tail": (res.test_output or "")[-700:],
        "seconds": round(time.perf_counter() - t0, 1),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker")
    args = ap.parse_args()
    out_dir = ROOT / "data" / "probes"
    config = EvalConfig(
        tasks_path=COMP / "tasks.jsonl",
        snapshots_dir=COMP / "snapshots",
        results_dir=out_dir / f"official_verify_{args.sandbox}_results",
        submission_dir=COMP / "sample_submission",
        models=ModelRegistry(),
        sandbox=args.sandbox,
        wheels_dir=COMP / "wheels",
    )
    ev = Evaluator(config)
    wanted = set(args.ids.split(","))
    tasks = [t for t in load_tasks(COMP / "tasks.jsonl") if t.instance_id in wanted]
    out = out_dir / f"official_verify_{args.sandbox}.jsonl"
    with out.open("a") as fh:
        for task in tasks:
            task = ev._hydrate_task_from_secret(task)
            gold = await grade(ev, task, task.patch)
            empty = await grade(ev, task, "")
            row = {
                "instance_id": task.instance_id,
                "sandbox": args.sandbox,
                "gold": gold,
                "empty": empty,
                "gradable": bool(gold["resolved"] and not empty["resolved"]),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                task.instance_id,
                "gold",
                gold["resolved"],
                gold["exit"],
                "empty",
                empty["resolved"],
                empty["exit"],
                (gold["error"] or "")[:80],
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())

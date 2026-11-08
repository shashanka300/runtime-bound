"""Gradability of the public tasks under the OFFICIAL harness (swegemma 0.2.7 from the scorer's
wheelhouse): for each task, run its tests at the base commit and with the reference (gold)
patch, using swegemma's own air-gapped verification (official dependency install, official
sandbox image). Compare with swebox's selfcheck (114/129 gradable) to find tasks the scorer's
environment cannot grade, which would cap every team's score.

    infra/official/.venv/bin/python tools/official_gold_sweep.py [--ids a,b] [--out ...]
Writes data/probes/official_gold.jsonl (one row per task).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from swegemma.harness.sample_verification import verify_sample_task_airgapped
from swegemma.models import load_tasks

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "data" / "competition"


def main() -> None:
    # verify_sample_task_airgapped looks for wheels at the relative path docker/wheels; the
    # scorer passes wheels_dir=<competition data>/wheels (getting-started notebook), so run from
    # a scratch cwd where docker/wheels points there.
    work = ROOT / "data" / "probes" / "official_cwd"
    (work / "docker").mkdir(parents=True, exist_ok=True)
    link = work / "docker" / "wheels"
    if not link.exists():
        link.symlink_to(COMP / "wheels")
    os.chdir(work)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "probes" / "official_gold.jsonl")
    args = ap.parse_args()
    tasks = load_tasks(COMP / "tasks.jsonl")
    if args.ids:
        wanted = set(args.ids.split(","))
        tasks = [t for t in tasks if t.instance_id in wanted]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {
            json.loads(line)["instance_id"] for line in args.out.read_text().splitlines() if line
        }
    with args.out.open("a") as fh:
        for task in tasks:
            if task.instance_id in done:
                continue
            t0 = time.monotonic()
            try:
                r = verify_sample_task_airgapped(
                    task=task, image="swebench-sandbox:latest", snapshots_dir=COMP / "snapshots"
                )
                row = {
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "base_exit": r.base_exit_code,
                    "patch_exit": r.patch_exit_code,
                    "gradable": r.passed_criteria,
                    "error": r.error,
                    "patch_tail": (r.patch_stdout or "")[-600:],
                }
            except Exception as exc:  # keep sweeping
                row = {
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "gradable": False,
                    "error": f"harness: {exc!r}"[:600],
                }
            row["seconds"] = round(time.monotonic() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                task.instance_id,
                row["gradable"],
                row.get("base_exit"),
                row.get("patch_exit"),
                row["seconds"],
                (row.get("error") or "")[:80],
                flush=True,
            )


if __name__ == "__main__":
    main()

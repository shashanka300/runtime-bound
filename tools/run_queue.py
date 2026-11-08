"""Overnight queue: run experiment notebooks headless with papermill, one after another.

    uv run python tools/run_queue.py queue.yaml [--dry-run]

queue.yaml:
    - experiment: E0_serving_benchmark
      overrides: {"params.concurrency": [1, 4]}
    - experiment: E1_starter_baseline
      overrides: {"seed": 1}

Each executed notebook (with outputs) is saved next to its run under
``archive_root()/queue/<stamp>/<idx>_<experiment>.ipynb`` (D: on this machine). A failure is logged and the
queue continues, so one bad job does not waste the night.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import papermill as pm
import yaml

from rbcore.paths import archive_root, experiments_dir


def run_queue(jobs: list[dict[str, Any]], dry_run: bool = False) -> list[dict[str, Any]]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = archive_root() / "queue" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for i, job in enumerate(jobs):
        exp = job["experiment"]
        exp_dir = experiments_dir() / exp
        nb = exp_dir / f"{exp}.ipynb"
        out = out_dir / f"{i:02d}_{exp}.ipynb"
        params = {"CONFIG_OVERRIDES": job.get("overrides", {})}
        print(f"[{i + 1}/{len(jobs)}] {exp} {params}")
        t0 = time.monotonic()
        status, error = "ok", None
        if not dry_run:
            try:
                pm.execute_notebook(
                    nb,
                    out,
                    parameters=params,
                    cwd=str(exp_dir),
                    kernel_name="challeenge",
                    progress_bar=False,
                )
            except Exception:
                status, error = "failed", traceback.format_exc(limit=3)
                print(error)
        report.append(
            {
                "experiment": exp,
                "overrides": params["CONFIG_OVERRIDES"],
                "status": status,
                "wall_s": round(time.monotonic() - t0, 1),
                "notebook": str(out),
                "error": error,
            }
        )
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = yaml.safe_load(args.queue.read_text())
    for r in run_queue(jobs, args.dry_run):
        print(f"{r['status']:>6}  {r['wall_s']:>8}s  {r['experiment']}")


if __name__ == "__main__":
    main()

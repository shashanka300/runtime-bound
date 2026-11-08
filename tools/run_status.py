"""Progress of a run (latest run of an experiment by default).

    uv run python tools/run_status.py E1_starter_baseline [--run <run_id>]

Reads only the run's files (events.jsonl, config.yaml, hw/, summary.json and, for swebox
experiments, the results directory), so it is safe to call while the run is in progress.
For a resumed run the results section counts the shared results directory, i.e. the
overall progress including results kept from earlier runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from rbcore.paths import experiments_dir
from rbcore.run import read_events, read_hw


def status(run_dir: Path) -> None:
    ev = read_events(run_dir)
    t0 = ev["t_ns"].min()
    now = time.time_ns()
    print(f"run      {run_dir.name}")
    summary = run_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        wall = f" in {s['wall_s'] / 60:.1f} min" if "wall_s" in s else ""
        print(f"status   {s['status']}{wall}")
    else:
        print(f"status   RUNNING for {(now - t0) / 6e10:.1f} min")

    starts = ev.filter(pl.col("event") == "phase_start")
    ends = ev.filter(pl.col("event") == "phase_end")
    for phase in starts["phase"].unique(maintain_order=True):
        n_start = starts.filter(pl.col("phase") == phase).height
        n_end = ends.filter(pl.col("phase") == phase).height
        line = f"phase    {phase:<10} {n_end}/{n_start} finished"
        if n_end < n_start:
            cur = starts.filter(pl.col("phase") == phase).row(-1, named=True)
            label = cur.get("key") or cur.get("instance_id") or cur.get("tag") or ""
            line += f", current {label} for {(now - cur['t_ns']) / 6e10:.1f} min"
        print(line)

    results_status(run_dir, ev, now)

    try:
        gpu = read_hw(run_dir, "gpu").tail(1).row(0, named=True)
        print(
            f"gpu      {gpu.get('util_gpu_pct')}% util, {gpu.get('power_w', 0):.0f} W, "
            f"{gpu.get('mem_used_mib', 0) / 1024:.1f} GiB, {gpu.get('temp_c')} C"
        )
    except FileNotFoundError:
        pass


def results_status(run_dir: Path, ev: pl.DataFrame, now: int) -> None:
    """Overall swebox progress per condition: done/expected, resolved, errors, ETA."""
    rd = ev.filter(pl.col("event") == "results_dir")
    root = (
        Path(rd["path"][-1]) if rd.height and "path" in rd.columns else run_dir / "artifacts/swebox"
    )
    tags = (
        sorted(p for p in root.glob("t*") if (p / "task_results.jsonl").exists())
        if root.exists()
        else []
    )
    if not tags and not root.exists():
        return
    params = _params(run_dir)
    per_tag = _expected_per_tag(params)
    order = [f"t{m:g}min" for m in params.get("time_budgets_min", [])]
    names = order or [p.name for p in tags]
    total_done = total_expected = 0
    print(f"results  {root}")
    for tag in names:
        f = root / tag / "task_results.jsonl"
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln] if f.exists() else []
        done = len(rows)
        resolved = sum(bool(r.get("resolved")) for r in rows)
        harness = sum(r.get("end_reason") == "harness_error" for r in rows)
        overflow = sum("ContextWindowExceeded" in str(r.get("agent_error")) for r in rows)
        exp = per_tag or done
        total_done += done
        total_expected += exp
        rate = f"{resolved / done:.1%}" if done else "-"
        print(
            f"  {tag:<7} {done:>4}/{exp:<4} resolved {resolved:>4} ({rate})  "
            f"context-overflow {overflow}  harness-errors {harness}"
        )
    # ETA from this run's own throughput (resumed runs skip finished work)
    task_ends = ev.filter((pl.col("event") == "phase_end") & (pl.col("phase") == "task"))
    elapsed_min = (now - ev["t_ns"].min()) / 6e10
    if task_ends.height and total_expected > total_done and elapsed_min > 0:
        per_min = task_ends.height / elapsed_min
        eta_h = (total_expected - total_done) / per_min / 60
        print(
            f"  overall {total_done}/{total_expected} "
            f"({total_done / total_expected:.1%}), ~{per_min * 60:.0f} tasks/h, "
            f"ETA ~{eta_h:.1f} h"
        )
    elif total_expected:
        print(f"  overall {total_done}/{total_expected}")


def _params(run_dir: Path) -> dict[str, Any]:
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return {}
    return dict((yaml.safe_load(cfg.read_text()) or {}).get("params") or {})


def _expected_per_tag(params: dict[str, Any]) -> int:
    if "subset" not in params or "seeds" not in params:
        return 0
    try:
        from swebox.tasks import load_tasks, select
    except ImportError:
        return 0
    return len(select(load_tasks(), params["subset"])) * len(params["seeds"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    runs = experiments_dir() / args.experiment / "runs"
    run_dir = runs / args.run if args.run else max(runs.iterdir(), key=lambda p: p.name)
    status(run_dir)


if __name__ == "__main__":
    main()

"""Move finished runs to cold storage, leaving a symlink so every path still resolves.

    uv run python tools/archive_runs.py --older-than-days 2 [--dry-run]

Only runs whose ``summary.json`` exists (i.e. the run closed) are moved. Analysis code
keeps working unchanged because ``experiments/<EX>/runs/<run_id>`` becomes a symlink
into ``archive_root()/runs_archive/<EX>/<run_id>``. Reads from the archive are
slower, so replay experiments that re-read a run heavily can copy it back first.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from rbcore.paths import archive_root, experiments_dir


def candidates(older_than_days: float) -> list[Path]:
    cutoff = time.time() - older_than_days * 86400
    return sorted(
        run
        for run in experiments_dir().glob("*/runs/*")
        if run.is_dir()
        and not run.is_symlink()
        and (run / "summary.json").exists()
        and (run / "summary.json").stat().st_mtime < cutoff
    )


def archive(run: Path, dry_run: bool = False) -> Path:
    dest = archive_root() / "runs_archive" / run.parent.parent.name / run.name
    if dry_run:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run), dest)
    run.symlink_to(dest, target_is_directory=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--older-than-days", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runs = candidates(args.older_than_days)
    for run in runs:
        print(f"{run}  ->  {archive(run, args.dry_run)}")
    print(f"{len(runs)} run(s) {'would be ' if args.dry_run else ''}archived")


if __name__ == "__main__":
    main()

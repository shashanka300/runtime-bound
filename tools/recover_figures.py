"""Restore every committed version of each figure into figures/runs/<name>__<run_id>.*
(one-off repair for figures saved under a fixed name before save_fig kept per-run copies).

    uv run python tools/recover_figures.py
Reads git history only; never modifies the current figures/<name>.* files.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> None:
    restored = 0
    for meta_path in sorted(ROOT.glob("experiments/*/figures/*.json")):
        rel = meta_path.relative_to(ROOT)
        stem = rel.with_suffix("")
        for sha in git("log", "--format=%H", "--", str(rel)).decode().split():
            try:
                meta = json.loads(git("show", f"{sha}:{rel}"))
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                continue
            runs = meta.get("source_runs") or []
            if not runs:
                continue
            archive = meta_path.parent / "runs"
            archive.mkdir(exist_ok=True)
            for ext in ("pdf", "png", "json"):
                dest = archive / f"{stem.name}__{runs[0]}.{ext}"
                if dest.exists():
                    continue
                try:
                    dest.write_bytes(git("show", f"{sha}:{stem}.{ext}"))
                    restored += 1
                except subprocess.CalledProcessError:
                    pass
    print(f"restored {restored} files")


if __name__ == "__main__":
    main()

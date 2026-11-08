"""Repair graph/embedding files left empty by hard-link-unaware unzipping.

In the Kaggle archive ``<instance_id>.{json,npz}`` and ``<repo>_<base_commit>.{json,npz}``
are hard links to each other (in either direction). Python's ``zipfile`` writes the
link side as an empty file; this re-links every empty copy to the copy that holds data.

    uv run python -m swebox.relink_code_data $DATA_ROOT/competition
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from swebox.tasks import load_tasks


def relink(comp: Path) -> int:
    fixed = 0
    for task in load_tasks(comp / "tasks.jsonl"):
        for kind, suffix in (("graphs", ".json"), ("embeddings", ".npz")):
            pair = (
                comp / kind / f"{task.instance_id}{suffix}",
                comp / kind / f"{task.repo_short}_{task.base_commit}{suffix}",
            )
            full = [p for p in pair if p.exists() and p.stat().st_size > 0]
            if not full:
                continue  # no data for this commit at all
            for p in pair:
                if not p.exists() or p.stat().st_size == 0:
                    p.unlink(missing_ok=True)
                    os.link(full[0], p)
                    fixed += 1
    return fixed


if __name__ == "__main__":
    print(f"relinked {relink(Path(sys.argv[1]))} files")

"""Benchmark tasks and the fixed, deterministic evaluation subsets used by the plan."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rbcore.paths import competition_dir

_DIFF_TARGET = re.compile(r"^\+\+\+ b/(\S+)", re.MULTILINE)


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""
    test_patch: str = ""
    created_at: str = ""

    @property
    def repo_short(self) -> str:
        return self.instance_id.rsplit("_", 1)[0]

    @cached_property
    def test_files(self) -> list[str]:
        """Files touched by ``test_patch``: reset before grading and passed to pytest."""
        return sorted(set(_DIFF_TARGET.findall(self.test_patch)))

    def snapshot(self, root: Path | None = None) -> Path:
        return (root or competition_dir()) / "snapshots" / f"{self.instance_id}.tgz"

    def _code_data(self, kind: str, suffix: str, root: Path | None) -> Path:
        """Task-named and commit-named files are hard links to each other in the Kaggle
        archive, in either direction; extractors that drop hard links leave one of the two
        empty. Return whichever copy holds the data."""
        base = (root or competition_dir()) / kind
        named = base / f"{self.instance_id}{suffix}"
        by_commit = base / f"{self.repo_short}_{self.base_commit}{suffix}"
        for p in (named, by_commit):
            if p.exists() and p.stat().st_size > 0:
                return p
        return named

    def graph_file(self, root: Path | None = None) -> Path:
        return self._code_data("graphs", ".json", root)

    def embeddings_file(self, root: Path | None = None) -> Path:
        return self._code_data("embeddings", ".npz", root)


def load_tasks(path: Path | None = None) -> list[Task]:
    path = path or competition_dir() / "tasks.jsonl"
    with path.open() as fh:
        return [Task.model_validate(json.loads(line)) for line in fh if line.strip()]


# -- subsets -----------------------------------------------------------------------------
# Stratified by repo and chosen by a stable hash of the instance id, so the subset never
# changes when tasks.jsonl is re-ordered and needs no RNG state.
SUBSET_QUOTAS: dict[str, dict[str, int]] = {
    "dev40": {"fastapi": 20, "rich": 15, "requests": 5},
    "fidelity10": {"fastapi": 5, "rich": 4, "requests": 1},
}


def _rank(instance_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{instance_id}".encode()).hexdigest()


def select(tasks: list[Task], subset: str | list[str]) -> list[Task]:
    """``subset`` is ``"all"``, a named subset (``dev40``, ``fidelity10``) or explicit ids."""
    if isinstance(subset, list):
        wanted = set(subset)
        missing = wanted - {t.instance_id for t in tasks}
        if missing:
            raise KeyError(f"unknown instance ids: {sorted(missing)}")
        return [t for t in tasks if t.instance_id in wanted]
    if subset == "all":
        return list(tasks)
    quotas = SUBSET_QUOTAS[subset]
    out: list[Task] = []
    for repo, n in quotas.items():
        pool = sorted(
            (t for t in tasks if t.repo_short == repo), key=lambda t: _rank(t.instance_id, subset)
        )
        out.extend(pool[:n])
    return sorted(out, key=lambda t: t.instance_id)

"""evaluate() scheduling: grading runs outside the agent slots; known slow graders go first."""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from swebox import evaluate as ev
from swebox import sandbox as sb
from swebox.agent_runner import AgentResult
from swebox.submission import ServingTarget
from swebox.tasks import Task
from swebox.tools import Budget
from swebox.verify import VerifyResult

AGENT_S, SLOW_GRADE_S = 0.2, 1.0


def _task(iid: str) -> Task:
    return Task(
        instance_id=iid, repo=f"o/{iid.rsplit('_', 1)[0]}", base_commit="x", problem_statement="p"
    )


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, float]]:
    log: list[tuple[str, str, float]] = []
    t0 = time.monotonic()

    async def fake_agent(dc: Any, task: Task, *_: Any) -> AgentResult:
        log.append(("agent_start", task.instance_id, time.monotonic() - t0))
        await asyncio.sleep(AGENT_S)
        return AgentResult("diff", True, "submitted", None, AGENT_S, 1, 1, 0)

    def fake_verify(dc: Any, task: Task, patch: str, limits: Any) -> VerifyResult:
        slow = task.instance_id in ev.SLOW_GRADING
        time.sleep(SLOW_GRADE_S if slow else 0.01)
        return VerifyResult(True, True, "git", 0, 0.0, "")

    for name, fn in (
        ("client", lambda: None),
        ("remove_orphans", lambda dc: 0),
        ("ensure_base_image", lambda dc: None),
        ("ensure_task_image", lambda dc, task, limits: None),
    ):
        monkeypatch.setattr(sb, name, fn)
    monkeypatch.setattr(ev, "run_agent", fake_agent)
    monkeypatch.setattr(ev, "verify", fake_verify)
    return log


def test_slow_grading_first_and_off_the_agent_slot(events: list, tmp_path: Path) -> None:
    slow = ev.SLOW_GRADING[0]
    tasks = [_task("fastapi_1"), _task("fastapi_2"), _task(slow), _task("fastapi_3")]
    spec = ev.EvalSpec(
        submission=tmp_path,
        budget=Budget(),
        limits=sb.Limits(1, 1),
        target=ServingTarget(),
        concurrency=1,
    )
    t = time.monotonic()
    rows = asyncio.run(ev.evaluate(tasks, spec, tmp_path / "out"))
    elapsed = time.monotonic() - t
    assert len(rows) == 4 and all(r["resolved"] for r in rows)
    starts = [iid for kind, iid, _ in events if kind == "agent_start"]
    assert starts[0] == slow  # scheduled first
    # 4 agents x 0.2 s serially; the 1 s grading overlaps them instead of adding to them
    assert elapsed < 4 * AGENT_S + SLOW_GRADE_S - 0.3

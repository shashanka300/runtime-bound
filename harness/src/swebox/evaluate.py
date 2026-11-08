"""Evaluate a submission over tasks x seeds, writing the spec's results layout (9.2)::

    <out>/summary.json  task_results.jsonl  patches/  test_outputs/  traces/  logs/

``traces/<instance>__s<seed>.jsonl`` holds the step-level trajectory (swebox.trace). When
a RunContext is given, each task is a telemetry phase: hardware samples, energy and
vLLM counters are attributed to tasks by ``rbcore.windows``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

from rbcore.run import RunContext
from swebox import sandbox as sb
from swebox.agent_runner import run_agent
from swebox.submission import ServingTarget
from swebox.tasks import Task
from swebox.tools import Budget
from swebox.verify import verify

log = structlog.get_logger(__name__)

SLOW_GRADING = ("rich_4006",)
"""Tasks whose hidden tests always run to the 30 min grading cap (rich_4006: 18/18 gradings
by 2026-09-27). Scheduled first so their grading overlaps other tasks' agent work instead of
trailing a run on an idle GPU. Patch-dependent hangs (e.g. rich_4076) are covered by grading
outside the agent slots."""


@dataclass(frozen=True)
class EvalSpec:
    submission: Path
    budget: Budget
    limits: sb.Limits
    target: ServingTarget
    seeds: tuple[int, ...] = (0,)
    concurrency: int = 2
    """Concurrent agent sessions (they share the model server)."""
    tag: str = ""
    """Free-form condition label (e.g. "t6min") stored on every result row."""
    verify_concurrency: int = 4
    """Concurrent gradings. Grading needs no model, so it runs outside the agent slots."""


class ResultsDir:
    def __init__(self, root: Path) -> None:
        self.root = root
        for sub in ("patches", "test_outputs", "traces", "logs"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.results = root / "task_results.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        with self.results.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.results.exists():
            return []
        return [json.loads(line) for line in self.results.read_text().splitlines() if line]


async def evaluate(
    tasks: Sequence[Task],
    spec: EvalSpec,
    out: Path,
    run: RunContext | None = None,
) -> list[dict[str, Any]]:
    res = ResultsDir(out)
    done = {(r["instance_id"], r["seed"], r.get("tag", "")) for r in res.rows()}  # resumable
    dc = sb.client()
    await asyncio.to_thread(sb.remove_orphans, dc)
    await asyncio.to_thread(sb.ensure_base_image, dc)
    agent_slots = asyncio.Semaphore(spec.concurrency)
    grade_slots = asyncio.Semaphore(spec.verify_concurrency)
    jobs = [(t, s) for s in spec.seeds for t in tasks if (t.instance_id, s, spec.tag) not in done]
    jobs.sort(key=lambda job: job[0].instance_id not in SLOW_GRADING)  # stable: slow first
    log.info("eval.start", jobs=len(jobs), skipped=len(done), concurrency=spec.concurrency)

    async def one(task: Task, seed: int) -> None:
        try:
            row = await _evaluate_one(dc, task, seed, spec, res, run, agent_slots, grade_slots)
        except Exception as exc:  # one broken task must not abort a long batch
            log.error("eval.task_harness_error", task=task.instance_id, error=repr(exc))
            row = {
                "instance_id": task.instance_id,
                "seed": seed,
                "tag": spec.tag,
                "repo": task.repo_short,
                "resolved": False,
                "end_reason": "harness_error",
                "agent_error": f"harness: {exc!r}"[:4000],
                "agent_seconds": None,
            }
        res.append(row)
        log.info(
            "eval.task_done",
            task=task.instance_id,
            seed=seed,
            resolved=row["resolved"],
            end=row["end_reason"],
            agent_s=row["agent_seconds"],
        )

    await asyncio.gather(*(one(t, s) for t, s in jobs))
    rows = [r for r in res.rows() if r.get("tag", "") == spec.tag]
    (out / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    return rows


async def _evaluate_one(
    dc: Any,
    task: Task,
    seed: int,
    spec: EvalSpec,
    res: ResultsDir,
    run: RunContext | None,
    agent_slots: asyncio.Semaphore,
    grade_slots: asyncio.Semaphore,
) -> dict[str, Any]:
    """Prep + agent run inside an agent slot (the telemetry "task" phase), then grading inside a
    grading slot, so a long grading (up to the 30 min cap) never holds an agent slot. Since
    2026-09-27 the "task" phase excludes grading, which uses no model."""
    key = f"{task.instance_id}__s{seed}"
    meta = {"instance_id": task.instance_id, "seed": seed, "tag": spec.tag}
    target = ServingTarget(**{**asdict(spec.target), "seed": seed})
    async with agent_slots:
        t0 = time.monotonic()
        phase = run.phase("task", **meta) if run else contextlib.nullcontext()
        with phase:
            prep = time.monotonic()
            await asyncio.to_thread(sb.ensure_task_image, dc, task, spec.limits)
            prep_s = time.monotonic() - prep
            agent = await run_agent(
                dc,
                task,
                spec.submission,
                target,
                spec.budget,
                spec.limits,
                res.root / "traces" / f"{key}.jsonl",
                meta,
            )
        (res.root / "patches" / f"{key}.patch").write_text(agent.patch)
    v = None
    if agent.patch.strip() or agent.agent_error is None:
        async with grade_slots:
            v = await asyncio.to_thread(verify, dc, task, agent.patch, spec.limits)
    if v is not None:
        (res.root / "test_outputs" / f"{key}.log").write_text(v.log)
    return {
        **meta,
        "repo": task.repo_short,
        "resolved": bool(v and v.resolved),
        "patch_applied": None if v is None else v.applied,
        "apply_method": None if v is None else v.apply_method,
        "test_exit_code": None if v is None else v.exit_code,
        "verify_error": None if v is None else v.error,
        "verify_seconds": None if v is None else v.seconds,
        "patch_chars": len(agent.patch),
        "patch_files": sum(1 for ln in agent.patch.splitlines() if ln.startswith("diff --git")),
        "submitted": agent.submitted,
        "end_reason": agent.end_reason,
        "agent_error": agent.agent_error,
        "agent_seconds": agent.agent_seconds,
        "prep_seconds": round(prep_s, 2),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "tool_calls": agent.tool_calls,
        "turns": agent.turns,
        "nudges": agent.nudges,
        "context_retries": agent.context_retries,
        "tool_counts": agent.tool_counts,
        **agent.usage,
    }


async def verify_only(
    tasks: Sequence[Task],
    which: str,
    limits: sb.Limits,
    out: Path,
    concurrency: int = 4,
    run: RunContext | None = None,
) -> list[dict[str, Any]]:
    """Harness self-check: grade the reference patch (``gold``, should resolve) or no patch
    (``empty``, should fail) for every task; no model involved."""
    dc = sb.client()
    await asyncio.to_thread(sb.ensure_base_image, dc)
    sem = asyncio.Semaphore(concurrency)
    logs = out / f"selfcheck_{which}"
    logs.mkdir(parents=True, exist_ok=True)

    async def one(task: Task) -> dict[str, Any]:
        async with sem:
            patch = task.patch if which == "gold" else ""
            with (
                run.phase("selfcheck", instance_id=task.instance_id, which=which)
                if run
                else contextlib.nullcontext()
            ):
                v = await asyncio.to_thread(verify, dc, task, patch, limits)
            (logs / f"{task.instance_id}.log").write_text(v.log)
            return {
                "instance_id": task.instance_id,
                "repo": task.repo_short,
                "which": which,
                "resolved": v.resolved,
                "applied": v.applied,
                "exit_code": v.exit_code,
                "error": v.error,
                "seconds": v.seconds,
            }

    return list(await asyncio.gather(*(one(t) for t in tasks)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    resolved = sum(r["resolved"] for r in rows)
    by_repo: dict[str, list[int]] = {}
    for r in rows:
        by_repo.setdefault(r["repo"], [0, 0])
        by_repo[r["repo"]][0] += int(r["resolved"])
        by_repo[r["repo"]][1] += 1
    return {
        "n": n,
        "resolved": resolved,
        "resolution_rate": resolved / n if n else None,
        "per_repo": {k: {"resolved": a, "n": b, "rate": a / b} for k, (a, b) in by_repo.items()},
        "errors": sum(1 for r in rows if r.get("agent_error")),
        "end_reasons": _count(r["end_reason"] for r in rows),
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out

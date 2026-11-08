"""Phase 1: run the compiled agent in Container A (spec sections 4, 5.3, 7.2, 8.1)."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
import structlog
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from swebox import graph as codegraph
from swebox.prompt import (
    LISTING_CMD,
    NUDGE_CONTINUE,
    NUDGE_MAX_TOKENS,
    NUDGE_TRUNCATED_TOOL_CALL,
    build_agent_prompt,
)
from swebox.sandbox import Limits, Sandbox, ensure_task_image
from swebox.submission import ServingTarget, compile_submission
from swebox.tasks import Task
from swebox.tools import Budget, TaskContext, create_tools
from swebox.trace import TrajectoryRecorder

log = structlog.get_logger(__name__)

MAX_NUDGES = 3
MIN_RETRY_SECONDS = 30.0
RETRY_NOTE = (
    "\n\n## Note\nA previous attempt on this task ran out of context. Any edits it made are "
    "still in /workspace: run `git diff` first and continue from there rather than starting "
    "over."
)
# Spec 7.2: compaction + context caching exactly as configured on the competition App.
COMPACTION = EventsCompactionConfig(
    compaction_interval=15, overlap_size=2, token_threshold=32_768, event_retention_size=5
)
CONTEXT_CACHE = ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)


@dataclass
class AgentResult:
    patch: str
    submitted: bool
    end_reason: str
    agent_error: str | None
    agent_seconds: float
    tool_calls: int
    turns: int
    nudges: int
    context_retries: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


async def run_agent(
    dc: docker.DockerClient,
    task: Task,
    submission: Path,
    target: ServingTarget,
    budget: Budget,
    limits: Limits,
    trace_file: Path,
    meta: dict[str, Any],
) -> AgentResult:
    image = await asyncio.to_thread(ensure_task_image, dc, task, limits)
    box = await asyncio.to_thread(
        Sandbox.start, dc, image, limits, "agent", task, {k: str(v) for k, v in meta.items()}
    )
    try:
        return await _session(box, task, submission, target, budget, trace_file, meta)
    finally:
        await asyncio.to_thread(box.remove)


async def _session(
    box: Sandbox,
    task: Task,
    submission: Path,
    target: ServingTarget,
    budget: Budget,
    trace_file: Path,
    meta: dict[str, Any],
) -> AgentResult:
    index = None
    if codegraph.available(task.graph_file(), task.embeddings_file()):
        index = await asyncio.to_thread(codegraph.load, task.graph_file(), task.embeddings_file())
    ctx = TaskContext(task=task, sandbox=box, budget=budget, code_index=index)
    compiled = compile_submission(submission, create_tools(ctx), target)
    recorder = TrajectoryRecorder(trace_file, ctx, meta)

    listing = (await asyncio.to_thread(box.exec, LISTING_CMD, 30)).stdout.strip()
    prompt = build_agent_prompt(task, budget, listing, code_intel=index is not None)

    app = App(
        name="swegemma_eval",
        root_agent=compiled.root,
        plugins=[recorder],
        events_compaction_config=COMPACTION,
        context_cache_config=CONTEXT_CACHE,
    )
    sessions = InMemorySessionService()
    state = {"problem_description": task.problem_statement}
    if task.hints_text.strip():
        state["hints"] = task.hints_text.strip()
    runner = Runner(app=app, session_service=sessions)

    ctx.start_agent_session()  # container setup is excluded from the time budget (spec 7.1)
    retries = 0
    try:
        session = await sessions.create_session(app_name=app.name, user_id="eval_user", state=state)
        end_reason, agent_error, nudges = await _converse(runner, session.id, prompt, ctx, task)
        # Diagnostic (not part of the official harness): recover from a context death by
        # continuing in a fresh session on the same workspace, edits preserved.
        while (
            retries < budget.context_retries
            and agent_error is not None
            and "ContextWindowExceeded" in agent_error
            and not ctx.patch_submitted
            and ctx.remaining_s > MIN_RETRY_SECONDS
        ):
            retries += 1
            log.info("agent.context_retry", task=task.instance_id, retry=retries)
            session = await sessions.create_session(
                app_name=app.name, user_id="eval_user", state=state
            )
            end_reason, agent_error, more = await _converse(
                runner, session.id, prompt + RETRY_NOTE, ctx, task
            )
            nudges += more
    finally:
        recorder.close_file()
    agent_seconds = ctx.elapsed_s

    # Spec 8.1: fall back to the working tree if the agent never called submit_patch.
    patch = ctx.submitted_patch if ctx.patch_submitted else await asyncio.to_thread(box.git_diff)
    return AgentResult(
        patch=patch,
        submitted=ctx.patch_submitted,
        end_reason=end_reason,
        agent_error=agent_error,
        agent_seconds=round(agent_seconds, 2),
        tool_calls=ctx.tool_calls_used,
        turns=ctx.turns_used,
        nudges=nudges,
        context_retries=retries,
        tool_counts=dict(ctx.tool_counts),
        usage=dict(recorder.totals),
    )


async def _converse(
    runner: Runner, session_id: str, prompt: str, ctx: TaskContext, task: Task
) -> tuple[str, str | None, int]:
    """Drive one session until submit / budget / nudges / error (spec 5.3).
    Returns (end_reason, agent_error, nudges)."""
    message, nudges, consecutive = prompt, 0, 0
    try:
        while True:
            turn_tool_call, last_text, finish = await _one_turn(runner, session_id, message, ctx)
            if ctx.patch_submitted:
                return "submitted", None, nudges
            if (exhausted := ctx.exhausted()) is not None:
                return exhausted, None, nudges
            consecutive = 0 if turn_tool_call else consecutive + 1
            if consecutive > MAX_NUDGES:
                return "nudges", None, nudges
            nudges += 1
            if "<|tool_call>" in last_text and "<tool_call|>" not in last_text:
                message = NUDGE_TRUNCATED_TOOL_CALL
            elif finish and ("MAX_TOKENS" in finish or "LENGTH" in finish.upper()):
                message = NUDGE_MAX_TOKENS
            else:
                message = NUDGE_CONTINUE
    except TimeoutError:
        return "time", None, nudges
    except Exception as exc:
        err = repr(exc)[:4000]
        log.warning("agent.error", task=task.instance_id, error=err)
        return "error", err, nudges


async def _one_turn(
    runner: Runner, session_id: str, message: str, ctx: TaskContext
) -> tuple[bool, str, str | None]:
    """One ``run_async`` invocation; returns (had tool call, last visible text, finish)."""
    had_tool_call, last_text, finish = False, "", None
    content = types.Content(role="user", parts=[types.Part(text=message)])
    events = runner.run_async(user_id="eval_user", session_id=session_id, new_message=content)
    # aclosing: when we break early, the generator is closed in this task's context, which
    # keeps ADK's OpenTelemetry context tokens valid ("created in a different Context").
    async with asyncio.timeout(max(1.0, ctx.remaining_s)), contextlib.aclosing(events):
        async for event in events:
            if event.get_function_calls():
                had_tool_call = True
            if event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts if not p.thought)
                if text:
                    last_text = text
            if event.finish_reason:
                finish = str(event.finish_reason)
            if ctx.patch_submitted and event.is_final_response():
                break
    return had_tool_call, last_text, finish

"""The nine built-in tools (spec section 6) bound to a per-task :class:`TaskContext`.

Tools are async (docker calls run in a worker thread) so concurrent tasks never block each
other's event loop. Every gated tool returns the spec's JSON envelope:
``{"status": "ok", ...}`` or ``{"status": "error", "error_type", "error_message", "details"}``.
"""

from __future__ import annotations

import asyncio
import difflib
import functools
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from swebox import graph as codegraph
from swebox.editing import EditError, apply_replacement
from swebox.sandbox import Sandbox
from swebox.tasks import Task

ToolFn = Callable[..., Awaitable[str]]
ToolImpl = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Budget:
    """``EvaluationBudget`` + ``HarnessLimits`` (spec 7.1)."""

    time_minutes: float = 60.0
    tool_calls: int | None = None
    turns: int = 500
    command_timeout_seconds: int = 300
    max_stdout_chars: int = 5000
    max_file_lines: int = 150
    max_file_chars: int = 10_000
    context_retries: int = 0
    """Diagnostic only (not in the official harness): fresh-session retries after a
    context-window rejection, on the same workspace."""


@dataclass
class TaskContext:
    task: Task
    sandbox: Sandbox
    budget: Budget
    code_index: codegraph.CodeIndex | None = None
    tool_calls_used: int = 0
    turns_used: int = 0
    patch_submitted: bool = False
    submitted_patch: str = ""
    agent_start: float | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)

    # -- time ------------------------------------------------------------------------------
    def start_agent_session(self) -> None:
        self.agent_start = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.agent_start is None else time.monotonic() - self.agent_start

    @property
    def remaining_s(self) -> float:
        return self.budget.time_minutes * 60 - self.elapsed_s

    @property
    def tool_calls_remaining(self) -> int | None:
        if self.budget.tool_calls is None:
            return None
        return max(0, self.budget.tool_calls - self.tool_calls_used)

    def exhausted(self) -> str | None:
        """Name of the exhausted budget, if any (ends the session)."""
        if self.remaining_s <= 0:
            return "time"
        if self.budget.tool_calls is not None and self.tool_calls_used >= self.budget.tool_calls:
            return "tool_calls"
        if self.turns_used >= self.budget.turns:
            return "turns"
        return None

    def status(self) -> dict[str, Any]:
        return {
            "tool_calls_used": self.tool_calls_used,
            "patch_submitted": self.patch_submitted,
            "patch_size": len(self.submitted_patch),
            "tool_calls_remaining": self.tool_calls_remaining,
            "max_tool_calls": self.budget.tool_calls,
            "time_seconds_remaining": round(max(0.0, self.remaining_s), 1),
            "max_time_minutes": self.budget.time_minutes,
            "agent_elapsed_seconds": round(self.elapsed_s, 1),
            "max_turns": self.budget.turns,
            "command_timeout_seconds": self.budget.command_timeout_seconds,
        }


# -- envelope helpers -----------------------------------------------------------------------
def _ok(**payload: Any) -> dict[str, Any]:
    return {"status": "ok", **payload}


def _err(kind: str, message: str, **details: Any) -> dict[str, Any]:
    return {"status": "error", "error_type": kind, "error_message": message, "details": details}


def _truncate(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
    )


def budget_gated(ctx: TaskContext, count_tool_call: bool = True) -> Callable[[ToolImpl], ToolFn]:
    def deco(fn: ToolImpl) -> ToolFn:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            ctx.tool_counts[fn.__name__] = ctx.tool_counts.get(fn.__name__, 0) + 1
            if ctx.remaining_s <= 0:
                out = _err("BudgetExhausted", "Session time budget exhausted.")
            elif (
                count_tool_call
                and ctx.budget.tool_calls is not None
                and ctx.tool_calls_used >= ctx.budget.tool_calls
            ):
                out = _err("BudgetExhausted", "Tool call budget exhausted. Call submit_patch.")
            else:
                if count_tool_call:
                    ctx.tool_calls_used += 1
                try:
                    out = await fn(*args, **kwargs)
                except Exception as exc:  # tool bugs must never kill the agent session
                    out = _err(type(exc).__name__, str(exc))
            remaining = ctx.tool_calls_remaining
            if (
                ctx.budget.tool_calls is not None
                and ctx.budget.tool_calls >= 20
                and remaining is not None
                and remaining <= 10
            ):
                out["budget_warning"] = (
                    f"Only {remaining} tool call(s) remaining ({ctx.tool_calls_used}/"
                    f"{ctx.budget.tool_calls} used). Finalize your edits and call submit_patch soon."
                )
            return json.dumps(out)

        return wrapper

    return deco


def _workspace_path(filepath: str) -> str:
    p = filepath.strip()
    for prefix in ("/workspace/", "/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
    if ".." in PurePosixPath(p).parts:
        raise ValueError(f"path traversal not allowed: {filepath!r}")
    if not p:
        raise ValueError("empty file path")
    return f"/workspace/{p}"


# -- tool factory ---------------------------------------------------------------------------
def create_tools(ctx: TaskContext) -> dict[str, ToolFn]:
    """Name -> async tool function. Docstrings become the tool descriptions for the model."""
    b = ctx.budget
    box = ctx.sandbox

    @budget_gated(ctx)
    async def run_command(command: str) -> dict[str, Any]:
        """Execute a bash command inside /workspace and return its stdout, stderr and exit code.

        Args:
            command: Shell command to run with /bin/bash -c.
        """
        timeout = min(b.command_timeout_seconds, max(5, int(ctx.remaining_s)))
        r = await asyncio.to_thread(box.exec, command, timeout)
        stdout = _truncate(r.stdout, b.max_stdout_chars)
        stderr = _truncate(r.stderr, b.max_stdout_chars)
        if r.timed_out:
            return _err(
                "TimeoutExceeded",
                f"Command timed out after {timeout} seconds",
                stdout=stdout,
                stderr=stderr,
                exit_code=r.exit_code,
            )
        if r.exit_code == 0:
            return _ok(stdout=stdout, exit_code=0)
        return _err(
            "CommandError",
            f"Command failed with exit code {r.exit_code}",
            stdout=stdout,
            stderr=stderr,
            exit_code=r.exit_code,
        )

    @budget_gated(ctx, count_tool_call=False)
    async def submit_patch() -> dict[str, Any]:
        """Capture all changes in /workspace as the final patch and end the task. Call last."""
        patch = await asyncio.to_thread(box.git_diff)
        ctx.submitted_patch = patch
        ctx.patch_submitted = True
        files = sum(1 for line in patch.splitlines() if line.startswith("diff --git"))
        return _ok(patch_size=len(patch), files_changed=files)

    async def get_status() -> str:
        """Return remaining budgets (time, tool calls, turns) and patch status. Free to call."""
        ctx.tool_counts["get_status"] = ctx.tool_counts.get("get_status", 0) + 1
        return json.dumps(ctx.status())

    @budget_gated(ctx)
    async def read_file(
        filepath: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict[str, Any]:
        """Read a file under /workspace with optional 1-indexed inclusive line range.

        Output is capped at 150 lines and 10000 characters per call.

        Args:
            filepath: Path relative to /workspace.
            start_line: First line to return (1-indexed, default 1).
            end_line: Last line to return (inclusive).
        """
        path = _workspace_path(filepath)
        text = (await asyncio.to_thread(box.read_bytes, path)).decode(errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(1, start_line or 1)
        want_end = min(total, end_line or total)
        out: list[str] = []
        chars, end = 0, start - 1
        for i in range(start, want_end + 1):
            line = lines[i - 1]
            if len(out) >= b.max_file_lines or chars + len(line) + 1 > b.max_file_chars:
                break
            out.append(line)
            chars += len(line) + 1
            end = i
        return _ok(
            filepath=path.removeprefix("/workspace/"),
            content="\n".join(out),
            start_line=start,
            end_line=end,
            total_lines=total,
            is_truncated=end < want_end,
        )

    @budget_gated(ctx)
    async def edit_file(
        filepath: str, old_string: str, new_string: str, allow_multiple: bool = False
    ) -> dict[str, Any]:
        """Replace old_string with new_string in an existing file under /workspace.

        Matching tries exact, then whitespace-flexible, then regex-tolerant matching.
        old_string must be unique unless allow_multiple is true.

        Args:
            filepath: Path relative to /workspace.
            old_string: Existing text to replace.
            new_string: Replacement text.
            allow_multiple: Replace every occurrence instead of requiring a unique match.
        """
        path = _workspace_path(filepath)
        try:
            before = (await asyncio.to_thread(box.read_bytes, path)).decode(errors="replace")
        except Exception:
            return _err("FileEditError", f"File not found: {filepath}")
        if not before:
            return _err("FileEditError", f"File is empty: {filepath}")
        try:
            rep = apply_replacement(before, old_string, new_string, allow_multiple)
        except EditError as exc:
            return _err("FileEditError", str(exc))
        await asyncio.to_thread(box.put_bytes, path, rep.content.encode())
        rel = path.removeprefix("/workspace/")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True), rep.content.splitlines(True), f"a/{rel}", f"b/{rel}"
            )
        )
        return _ok(
            filepath=rel,
            occurrences=rep.occurrences,
            strategy=rep.strategy,
            diff=_truncate(diff, b.max_stdout_chars),
            is_truncated=len(diff) > b.max_stdout_chars,
        )

    @budget_gated(ctx)
    async def write_file(filepath: str, content: str) -> dict[str, Any]:
        """Create or overwrite a file under /workspace (parent directories are created).

        Args:
            filepath: Path relative to /workspace.
            content: Full file content.
        """
        path = _workspace_path(filepath)
        data = content.encode()
        await asyncio.to_thread(box.put_bytes, path, data)
        return _ok(filepath=path.removeprefix("/workspace/"), size=len(data))

    tools: dict[str, ToolFn] = {
        "run_command": run_command,
        "submit_patch": submit_patch,
        "get_status": get_status,
        "read_file": read_file,
        "edit_file": edit_file,
        "write_file": write_file,
    }
    idx = ctx.code_index
    unavailable = _err(
        "GraphUnavailable", "No pre-computed code graph/embeddings for this repository."
    )
    # Graph tools: always registered (static tool registry, as in swegemma).

    @budget_gated(ctx)
    async def get_code_neighbors(
        node: str, edge_type: str | None = None, max_neighbors: int = 50
    ) -> dict[str, Any]:
        """Find callers, callees and related symbols of a symbol in the code graph.

        Args:
            node: Symbol name, e.g. "FastAPI.get" or "Request".
            edge_type: Optional edge filter such as "CALLS".
            max_neighbors: Maximum number of neighbors to return.
        """
        if idx is None:
            return dict(unavailable)
        return _ok(**idx.neighbors(node, edge_type, max_neighbors))

    @budget_gated(ctx)
    async def search_similar_code(query: str, k: int = 10) -> dict[str, Any]:
        """Find the k functions/classes most similar to a symbol (pass a symbol name).

        Args:
            query: A class, function or module symbol name, not free text.
            k: Number of results.
        """
        if idx is None:
            return dict(unavailable)
        return _ok(**idx.similar(query, k))

    @budget_gated(ctx)
    async def get_code_subgraph(nodes: list[str]) -> dict[str, Any]:
        """Return the induced call subgraph (nodes and edges) for a list of symbols.

        Args:
            nodes: Symbol names.
        """
        if idx is None:
            return dict(unavailable)
        return _ok(**idx.subgraph(nodes))

    tools |= {
        "get_code_neighbors": get_code_neighbors,
        "search_similar_code": search_similar_code,
        "get_code_subgraph": get_code_subgraph,
    }
    return tools


GRAPH_TOOLS = frozenset({"get_code_neighbors", "search_similar_code", "get_code_subgraph"})

"""Step-level trajectory recorder: an ADK plugin writing one JSON line per model call,
tool call and context-compaction event.

This is the raw material for token accounting (E2): what went into every prompt, broken
down by kind (instructions, user text, model text, thoughts, tool calls, tool
outputs), what came back, how much of the prompt the prefix cache served, and which files
and line ranges each read touched (re-read detection). The plugin also enforces the
``turns`` budget.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import IO, Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from swebox.tools import TaskContext

ARG_CHARS = 2000  # stored tool-argument text per call (full args of edits are not needed)


def _content_breakdown(contents: list[types.Content]) -> dict[str, int]:
    """Characters in the request by kind: the prompt-side token accounting input."""
    out = {"user_text": 0, "model_text": 0, "thought": 0, "tool_call": 0, "tool_output": 0}
    for c in contents:
        for p in c.parts or []:
            if p.function_call is not None:
                out["tool_call"] += len(json.dumps(p.function_call.args or {}, default=str))
            elif p.function_response is not None:
                out["tool_output"] += len(
                    json.dumps(p.function_response.response or {}, default=str)
                )
            elif p.text:
                key = (
                    "thought" if p.thought else ("model_text" if c.role == "model" else "user_text")
                )
                out[key] += len(p.text)
    return out


class TrajectoryRecorder(BasePlugin):
    def __init__(self, path: Path, ctx: TaskContext, run_meta: dict[str, Any]) -> None:
        super().__init__(name="swebox_trajectory")
        self._fh: IO[str] = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.ctx = ctx
        self.meta = run_meta
        self.step = 0
        self._model_t0: dict[str, float] = {}
        self._tool_t0: dict[str, float] = {}
        self.totals = {
            "model_calls": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "thought_tokens": 0,
            "compactions": 0,
            "model_errors": 0,
        }

    def _write(self, kind: str, **rec: Any) -> None:
        self.step += 1
        line = {
            "step": self.step,
            "kind": kind,
            "t_ns": time.time_ns(),
            "elapsed_s": round(self.ctx.elapsed_s, 3),
            **self.meta,
            **rec,
        }
        with self._lock:
            self._fh.write(json.dumps(line, default=str) + "\n")
            self._fh.flush()

    def close_file(self) -> None:
        self._fh.close()

    # -- model calls ---------------------------------------------------------------------
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        self.ctx.turns_used += 1
        if self.ctx.turns_used > self.ctx.budget.turns:
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="Turn budget exhausted.")]
                )
            )
        self._model_t0[callback_context.invocation_id + callback_context.agent_name] = (
            time.monotonic()
        )
        cfg = llm_request.config
        sys_chars = len(str(cfg.system_instruction)) if cfg and cfg.system_instruction else 0
        self._write(
            "model_request",
            agent=callback_context.agent_name,
            invocation=callback_context.invocation_id,
            n_contents=len(llm_request.contents),
            n_tools=len(llm_request.tools_dict),
            system_chars=sys_chars,
            chars=_content_breakdown(llm_request.contents),
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        key = callback_context.invocation_id + callback_context.agent_name
        t0 = self._model_t0.pop(key, None)
        u = llm_response.usage_metadata
        usage = {
            "prompt_tokens": getattr(u, "prompt_token_count", None) or 0,
            "cached_tokens": getattr(u, "cached_content_token_count", None) or 0,
            "completion_tokens": getattr(u, "candidates_token_count", None) or 0,
            "thought_tokens": getattr(u, "thoughts_token_count", None) or 0,
        }
        if u is not None:
            self.totals["model_calls"] += 1
            for k, v in usage.items():
                self.totals[k] += v
        parts = llm_response.content.parts if llm_response.content else []
        calls = [p.function_call.name for p in parts or [] if p.function_call]
        text = "".join(p.text or "" for p in parts or [] if not p.thought)
        self._write(
            "model_response",
            agent=callback_context.agent_name,
            invocation=callback_context.invocation_id,
            latency_s=None if t0 is None else round(time.monotonic() - t0, 3),
            finish_reason=str(llm_response.finish_reason) if llm_response.finish_reason else None,
            tool_calls=calls,
            text_chars=len(text),
            unclosed_tool_call="<|tool_call>" in text and "<tool_call|>" not in text,
            error=llm_response.error_code,
            partial=llm_response.partial,
            **usage,
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> LlmResponse | None:
        self.totals["model_errors"] += 1
        self._write("model_error", agent=callback_context.agent_name, error=repr(error)[:2000])
        return None

    # -- tool calls ----------------------------------------------------------------------
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        self._tool_t0[tool_context.function_call_id or tool.name] = time.monotonic()
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        t0 = self._tool_t0.pop(tool_context.function_call_id or tool.name, None)
        raw = result.get("result", result) if isinstance(result, dict) else result
        status, error_type, extra = None, None, {}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                status, error_type = parsed.get("status"), parsed.get("error_type")
                if tool.name == "read_file" and status == "ok":
                    extra = {
                        k: parsed.get(k)
                        for k in (
                            "filepath",
                            "start_line",
                            "end_line",
                            "total_lines",
                            "is_truncated",
                        )
                    }
        except (TypeError, ValueError):
            pass
        self._write(
            "tool",
            tool=tool.name,
            agent=tool_context.agent_name,
            args=json.dumps(tool_args, default=str)[:ARG_CHARS],
            output_chars=len(raw) if isinstance(raw, str) else len(json.dumps(raw, default=str)),
            status=status,
            error_type=error_type,
            latency_s=None if t0 is None else round(time.monotonic() - t0, 3),
            **extra,
        )
        return None

    # -- events (compaction) -------------------------------------------------------------
    async def on_event_callback(self, *, invocation_context: Any, event: Event) -> Event | None:
        compaction = getattr(event.actions, "compaction", None) if event.actions else None
        if compaction is not None:
            self.totals["compactions"] += 1
            self._write("compaction", author=event.author, detail=str(compaction)[:500])
        return None

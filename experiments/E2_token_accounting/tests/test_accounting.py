"""Token accounting on a hand-built trace whose answers are known."""

import polars as pl
import pytest
from src import accounting as acc

KEY = {"tag": "t6min", "instance_id": "repo_1", "seed": 0}
AGENT = {"agent": "root", "invocation": "inv1"}


def _req(step: int, tool_output: int = 0, thought: int = 0) -> dict[str, object]:
    return {
        **KEY,
        **AGENT,
        "step": step,
        "kind": "model_request",
        "t_ns": step,
        "n_contents": step,
        "system_chars": 1000,
        "user_text": 1000,
        "model_text": 0,
        "thought": thought,
        "tool_call": 0,
        "tool_output": tool_output,
    }


def _resp(step: int, prompt: int, cached: int, out: int = 50, think: int = 20) -> dict[str, object]:
    return {
        **KEY,
        **AGENT,
        "step": step,
        "kind": "model_response",
        "t_ns": step,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": out,
        "thought_tokens": think,
    }


def _tool(step: int, tool: str, chars: int, args: str, **extra: object) -> dict[str, object]:
    return {
        **KEY,
        "agent": "root",
        "invocation": None,
        "step": step,
        "kind": "tool",
        "t_ns": step,
        "tool": tool,
        "output_chars": chars,
        "args": args,
        **extra,
    }


@pytest.fixture
def steps() -> pl.DataFrame:
    rows = [
        _req(1),
        _resp(2, prompt=1000, cached=0),
        _tool(
            3,
            "read_file",
            400,
            '{"filepath": "a.py"}',
            filepath="a.py",
            start_line=1,
            end_line=150,
            total_lines=300,
            is_truncated=True,
        ),
        _req(4, tool_output=400),
        _resp(5, prompt=1200, cached=1000),
        _tool(6, "run_command", 400, '{"command": "sed -n \'1,100p\' a.py"}'),
        # cache evicted: only 200 of the prefix still cached -> 800 recomputed
        _req(7, tool_output=800),
        _resp(8, prompt=1400, cached=200),
        {
            **KEY,
            **AGENT,
            "step": 9,
            "kind": "model_request",
            "t_ns": 9,
            "n_contents": 9,
            "system_chars": 1000,
            "user_text": 1000,
            "model_text": 0,
            "thought": 0,
            "tool_call": 0,
            "tool_output": 90000,
        },
        {
            **KEY,
            "agent": "root",
            "invocation": None,
            "step": 10,
            "kind": "model_error",
            "t_ns": 10,
            "error": "ContextWindowExceededError",
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def test_calls_and_compute_view(steps: pl.DataFrame) -> None:
    calls = acc.model_calls(steps)
    assert calls.height == 4 and calls["overflow"].to_list() == [False, False, False, True]
    cv = acc.compute_view(calls)
    assert cv["new"].to_list() == [1000, 200, 200]
    assert cv["recomputed"].to_list() == [0, 0, 1000]  # uncached 1200 - new 200


def test_segments_reread_truncation_lifetime(steps: pl.DataFrame) -> None:
    calls = acc.model_calls(steps)
    fits = {"root": acc.CharFit("root", 0.25, 0.0, 1.0, 3)}
    seg = acc.segments(steps, calls, fits, reread_overlap=0.5, oversized_chars=20000)
    first, second = seg.sort("step").iter_rows(named=True)
    assert first["truncated"] and not first["reread"] and first["lifetime"] == 2
    assert second["reread"] and second["lifetime"] == 1  # sed 1-100 inside the 1-150 read
    assert second["token_steps"] == pytest.approx(100.0)  # 400 chars * 0.25 * 1 call


def test_per_task_bounds(steps: pl.DataFrame) -> None:
    calls = acc.model_calls(steps)
    fits = {"root": acc.CharFit("root", 0.25, 0.0, 1.0, 3)}
    seg = acc.segments(steps, calls, fits, 0.5, 20000)
    t = acc.per_task(acc.compute_view(calls), seg, calls, fits).row(0, named=True)
    total = 1000 + 1200 + 1400 + 3 * 50
    assert t["total_tokens"] == total
    assert t["overflow_calls"] == 1
    assert t["waste_share_upper"] == pytest.approx((100 + 200 + 1000) / total)
    assert t["waste_share_lower"] == pytest.approx(1000 / total)


def test_read_ranges_parsing() -> None:
    assert acc.read_ranges("run_command", '{"command": "cat src/x.py"}', None, None, None) == [
        ("src/x.py", 1, 10**9)
    ]
    assert acc.read_ranges(
        "run_command", '{"command": "sed -n 5,9p /workspace/y.py"}', None, None, None
    ) == [("y.py", 5, 9)]
    assert acc.read_ranges("run_command", '{"command": "grep -n foo x.py"}', None, None, None) == []

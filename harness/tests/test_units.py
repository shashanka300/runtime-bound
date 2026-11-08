"""Unit tests that need neither Docker nor a model server."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from rbcore.paths import competition_dir
from swebox import graph as codegraph
from swebox.editing import EditError, apply_replacement
from swebox.prompt import build_agent_prompt
from swebox.submission import (
    ServingTarget,
    SubmissionError,
    compile_submission,
    official_thinking_body,
    thinking_extra_body,
)
from swebox.tasks import Task, load_tasks, select
from swebox.tools import Budget

HAVE_DATA = (competition_dir() / "tasks.jsonl").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="competition data not downloaded")


# -- editing -------------------------------------------------------------------------------
def test_exact_then_flexible_then_regex() -> None:
    src = "def f(a, b):\n    return a+b\n"
    assert apply_replacement(src, "a+b", "a - b").strategy == "exact"
    flex = apply_replacement(
        "class A:\n    def f(self):\n        x = 1\n        return x\n",
        "def f(self):\n    x = 1",
        "def f(self):\n    x = 2",
    )
    assert flex.strategy == "flexible"
    assert "        x = 2" in flex.content  # re-indented to the matched block
    rx = apply_replacement("call(a,\n     b)\n", "call( a, b )", "call(b, a)")
    assert rx.strategy == "regex" and rx.content.startswith("call(b, a)")


def test_edit_uniqueness_and_errors() -> None:
    with pytest.raises(EditError):
        apply_replacement("x\nx\n", "x", "y")
    assert apply_replacement("x\nx\n", "x", "y", allow_multiple=True).occurrences == 2
    with pytest.raises(EditError):
        apply_replacement("abc", "", "d")
    with pytest.raises(EditError):
        apply_replacement("abc", "zzz", "d")


# -- tasks & subsets -------------------------------------------------------------------------
def test_test_files_parsed_from_patch() -> None:
    t = Task(
        instance_id="r_1",
        repo="o/r",
        base_commit="0" * 40,
        problem_statement="p",
        test_patch="--- a/tests/a.py\n+++ b/tests/a.py\n@@\n--- a/t/b.py\n+++ b/t/b.py\n",
    )
    assert t.test_files == ["t/b.py", "tests/a.py"]


@needs_data
def test_subsets_are_stratified_and_stable() -> None:
    tasks = load_tasks()
    dev = select(tasks, "dev40")
    assert len(dev) == 40
    counts = {r: sum(t.repo_short == r for t in dev) for r in ("fastapi", "rich", "requests")}
    assert counts == {"fastapi": 20, "rich": 15, "requests": 5}
    assert [t.instance_id for t in select(list(reversed(tasks)), "dev40")] == [
        t.instance_id for t in dev
    ]
    fid = {t.instance_id for t in select(tasks, "fidelity10")}
    assert len(fid) == 10


# -- code graph --------------------------------------------------------------------------------
def _index() -> codegraph.CodeIndex:
    g = nx.MultiDiGraph()
    for n in ("pkg.mod.Request", "pkg.mod.Request.send", "pkg.util.parse"):
        g.add_node(n, text=f"def {n}(): ...")
    g.add_edge("pkg.mod.Request.send", "pkg.util.parse", type="calls")
    vec = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return codegraph.CodeIndex(g, list(g.nodes), vec)


def test_graph_resolution_neighbors_similarity() -> None:
    idx = _index()
    assert idx.resolve("Request") == "pkg.mod.Request"  # suffix tier
    assert idx.resolve("PKG.UTIL.PARSE") == "pkg.util.parse"  # case-insensitive tier
    assert idx.resolve("util") == "pkg.util.parse"  # substring tier
    nb = idx.neighbors("send", "CALLS", 50)
    assert nb["neighbors"] == ["pkg.util.parse"]
    sim = idx.similar("Request", 1)
    assert sim["results"][0]["node_name"] == "pkg.mod.Request.send"
    sub = idx.subgraph(["send", "parse"])
    assert sub["edge_count"] == 1 and sub["edges"][0]["type"] == "CALLS"
    with pytest.raises(KeyError):
        idx.neighbors("nope_zzz", None, 5)


# -- submission ----------------------------------------------------------------------------------
def test_official_thinking_mapping_matches_adk_submission() -> None:
    # adk_submission 0.2.11 resolvers/generation.py: no thinking budget is ever sent
    starter = {"thinking_level": "high", "thinking_budget": 4096, "include_thoughts": True}
    assert official_thinking_body(starter) == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "high",
    }
    no_level = {"thinking_budget": 4096, "include_thoughts": True}
    assert official_thinking_body(no_level) == {"chat_template_kwargs": {"enable_thinking": True}}
    off = official_thinking_body({"thinking_level": "NONE", "include_thoughts": False})
    assert off == {"chat_template_kwargs": {"enable_thinking": False}}
    assert official_thinking_body({"thinking_level": "minimal"}) == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    # wheel of 2026-09-27: a non-positive thinking_budget switches thinking off
    assert official_thinking_body({"thinking_level": "high", "thinking_budget": 0}) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_thinking_mapping() -> None:
    assert thinking_extra_body({}) == {}
    on = thinking_extra_body({"thinking_level": "high", "thinking_budget": 4096})
    assert on == {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}
    off = thinking_extra_body({"thinking_level": "NONE"})
    assert off == {"chat_template_kwargs": {"enable_thinking": False}}


async def _noop(**_: object) -> str:
    return "{}"


@needs_data
def test_compile_starter_submission() -> None:
    names = [
        "run_command",
        "read_file",
        "edit_file",
        "write_file",
        "get_status",
        "submit_patch",
        "get_code_neighbors",
        "search_similar_code",
        "get_code_subgraph",
    ]
    tools = {n: _noop for n in names}
    compiled = compile_submission(
        competition_dir() / "sample_submission", tools, ServingTarget(seed=3)
    )
    assert compiled.model == "gemma-4-31b-it-qat-w4a16-ct"
    assert compiled.adapters == ["main_lora", "tool_lora"]
    assert compiled.agent_count == 2
    root = compiled.root
    assert root.generate_content_config.seed == 3  # type: ignore[attr-defined]


def test_compile_rejects_escape_and_two_models(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text(
        "name: a\nmodel: gemma-4-31b-it\ninstruction: !include ../x.md\n"
    )
    (tmp_path.parent / "x.md").write_text("hi")
    with pytest.raises(SubmissionError):
        compile_submission(tmp_path, {}, ServingTarget())
    (tmp_path / "agent.yaml").write_text(
        "name: a\nmodel: gemma-4-31b-it\nsub_agents:\n  - {name: b, model: gemma-4-12b-it}\n"
    )
    with pytest.raises(SubmissionError, match="single base model"):
        compile_submission(tmp_path, {}, ServingTarget())


# -- prompt ---------------------------------------------------------------------------------------
def test_prompt_sections() -> None:
    t = Task(
        instance_id="r_1",
        repo="o/r",
        base_commit="0" * 40,
        problem_statement="Bug X",
        hints_text="look at y",
    )
    p = build_agent_prompt(t, Budget(time_minutes=6, tool_calls=40), "./a\n./b", code_intel=True)
    for section in (
        "Problem Statement:\nBug X",
        "## Hints:",
        "- Time allowance: 6.0 minutes",
        "- Tool calls allowance: 40 calls",
        "## Execution Environment Rules",
        "## Code Intelligence Tools",
        "## Workspace Layout",
    ):
        assert section in p
    assert "Code Intelligence" not in build_agent_prompt(t, Budget(), "", code_intel=False)

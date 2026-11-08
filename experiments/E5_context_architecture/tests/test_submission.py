import re
from pathlib import Path

import pytest
from src.build_submission import PARTS, VARIANTS, build, phase_prompt, phases

from rbcore.paths import competition_dir
from swebox.submission import ServingTarget, compile_submission

NEEDS_DATA = pytest.mark.skipif(
    not (competition_dir() / "sample_submission").exists(), reason="competition data not downloaded"
)
ALL_TOOLS = [
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
ALLOWED_KEYS = {
    "problem_description",
    "hints?",
    "localization",
    "repro",
    "verification?",
    "edit_summary",
}


async def _stub(**_: object) -> str:
    return ""


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prompts_share_prefix_and_template_only_state_keys(variant: str) -> None:
    shared = (PARTS / VARIANTS[variant].prompts / "shared.md").read_text()
    for phase in phases(VARIANTS[variant]):
        text = phase_prompt(variant, phase)
        assert text.startswith(shared)
        keys = set(re.findall(r"\{([^{}]*)\}", text))
        assert keys <= ALLOWED_KEYS, (variant, phase, keys - ALLOWED_KEYS)


@NEEDS_DATA
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_compiles(variant: str, tmp_path: Path) -> None:
    v = VARIANTS[variant]
    out = build(variant, tmp_path / variant)
    compiled = compile_submission(out, dict.fromkeys(ALL_TOOLS, _stub), ServingTarget())
    root = compiled.root
    names = [a.name for a in root.sub_agents]
    assert names == [f"{p}_agent" for p in v.pipeline] + ["fix_loop"]
    loop = root.sub_agents[-1]
    assert loop.max_iterations == v.max_fix_iterations
    agents = [*root.sub_agents[:-1], *loop.sub_agents]
    for agent in agents:
        assert agent.include_contents == "none"
    localizer = root.sub_agents[0]
    assert not {"edit_file", "write_file"} & {t.name for t in localizer.tools}
    budget = v.thinking.get("localizer")
    if budget is not None:
        body = localizer.generate_content_config.http_options.extra_body
        assert body["chat_template_kwargs"]["enable_thinking"] is (budget > 0)
        if budget:
            assert body["thinking_token_budget"] == budget


@NEEDS_DATA
def test_sampling_overrides_apply_per_phase(tmp_path: Path) -> None:
    out = build("phase_v2_3", tmp_path / "v23")
    root = compile_submission(out, dict.fromkeys(ALL_TOOLS, _stub), ServingTarget()).root
    localizer, loop = root.sub_agents
    editor, verifier = loop.sub_agents
    for agent in (localizer, editor):
        assert agent.generate_content_config.temperature == 1.0
        assert agent.generate_content_config.top_k == 64
    assert verifier.generate_content_config.temperature == 0.2
    extra = localizer.generate_content_config.http_options.extra_body
    assert extra["chat_template_kwargs"]["enable_thinking"] is False  # override keeps thinking off
    assert editor.generate_content_config.http_options.extra_body["thinking_token_budget"] == 4096


def test_v2_drops_reproducer_and_get_status() -> None:
    v = VARIANTS["phase_v2"]
    assert "reproducer" not in phases(v)
    assert all("get_status" not in tools for tools in v.tools.values())

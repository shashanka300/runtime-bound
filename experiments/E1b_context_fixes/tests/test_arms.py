from pathlib import Path

import pytest
import yaml
from src.arms import GUIDANCE, context_retries, make_arm


@pytest.fixture
def base(tmp_path: Path) -> Path:
    b = tmp_path / "base"
    (b / "configs").mkdir(parents=True)
    (b / "prompts").mkdir()
    (b / "configs" / "sampling.yaml").write_text(
        "temperature: 0.2\nmax_output_tokens: 16384\nthinking_config:\n  thinking_budget: 4096\n"
    )
    (b / "prompts" / "system.md").write_text("SYSTEM")
    return b


def test_arm_a_is_unchanged(base: Path, tmp_path: Path) -> None:
    a = make_arm(base, tmp_path / "a", "A", 4096)
    assert (a / "configs" / "sampling.yaml").read_text() == (
        base / "configs" / "sampling.yaml"
    ).read_text()


def test_arm_b_sets_reserve_only(base: Path, tmp_path: Path) -> None:
    b = make_arm(base, tmp_path / "b", "B", 4096)
    cfg = yaml.safe_load((b / "configs" / "sampling.yaml").read_text())
    assert cfg["max_output_tokens"] == 8192 and cfg["temperature"] == 0.2
    assert (b / "prompts" / "system.md").read_text() == "SYSTEM"


def test_arm_c_adds_guidance_on_top_of_b(base: Path, tmp_path: Path) -> None:
    c = make_arm(base, tmp_path / "c", "C", 4096)
    assert (c / "prompts" / "system.md").read_text() == "SYSTEM" + GUIDANCE
    assert (
        yaml.safe_load((c / "configs" / "sampling.yaml").read_text())["max_output_tokens"] == 8192
    )


def test_arm_e_is_b_plus_gemma_sampling(base: Path, tmp_path: Path) -> None:
    e = make_arm(base, tmp_path / "e", "E", 4096)
    cfg = yaml.safe_load((e / "configs" / "sampling.yaml").read_text())
    assert cfg == {
        "temperature": 1.0,
        "max_output_tokens": 8192,
        "thinking_config": {"thinking_budget": 4096},
        "top_p": 0.95,
        "top_k": 64,
    }
    assert (e / "prompts" / "system.md").read_text() == "SYSTEM"


def test_arm_f_is_b_without_adapters(base: Path, tmp_path: Path) -> None:
    (base / "adapters" / "main_lora").mkdir(parents=True)
    (base / "adapters" / "main_lora" / "adapter_config.json").write_text("{}")
    (base / "sub_agents").mkdir()
    (base / "agent.yaml").write_text(
        "name: a\nmodel: m\nadapter: main_lora\ninstruction: !include prompts/system.md\n"
    )
    (base / "sub_agents" / "s.yaml").write_text("name: s\n  adapter: tool_lora\nmodel: m\n")
    f = make_arm(base, tmp_path / "f", "F", 4096)
    assert not (f / "adapters").exists()
    assert (f / "agent.yaml").read_text() == (
        "name: a\nmodel: m\ninstruction: !include prompts/system.md\n"
    )
    assert "adapter" not in (f / "sub_agents" / "s.yaml").read_text()
    b = yaml.safe_load((f / "configs" / "sampling.yaml").read_text())
    assert b["max_output_tokens"] == 8192 and b["temperature"] == 0.2  # otherwise identical to B


def test_arms_g_h_bound_thinking_through_max_output(base: Path, tmp_path: Path) -> None:
    (base / "adapters" / "main_lora").mkdir(parents=True)
    g = make_arm(base, tmp_path / "g", "G", 4096)
    cfg = yaml.safe_load((g / "configs" / "sampling.yaml").read_text())
    assert cfg["max_output_tokens"] == 4096
    assert cfg["thinking_config"] == {"include_thoughts": False}
    assert not (g / "adapters").exists()
    h = make_arm(base, tmp_path / "h", "H", 4096)
    cfg = yaml.safe_load((h / "configs" / "sampling.yaml").read_text())
    assert cfg["max_output_tokens"] == 4096
    assert cfg["thinking_config"] == {"thinking_budget": 4096}  # thinking stays on
    assert not (h / "adapters").exists()


def test_only_d_retries() -> None:
    assert context_retries("D", 2) == 2 and context_retries("B", 2) == 0

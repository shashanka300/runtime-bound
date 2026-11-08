"""Generate declarative phase-agent submissions from prompt parts, one named variant each.

Submissions are YAML-only, and an agent's ``instruction`` is one string, so each phase prompt
is written as ``<variant>/shared.md + <variant>/<phase>.md``. Generating it guarantees the
shared block is byte-identical across phases (a common token prefix for prefix caching).

Variants (every one stays reproducible; results record which was run):
- ``phase_v1``: locate -> reproduce -> (edit -> verify) x3, thinking 4096 everywhere.
- ``phase_v2``: from the v1 smoke test (65% of time in localizer model calls, ~7 s each;
  reproducer ~80 s/task; editor reached in 5/10 tasks): no reproducer, localizer thinking
  off and capped at 8 calls in its prompt, get_status removed, fix loop x2.
- ``phase_v2_1``: from the v2 smoke test (one localizer, thinking off, repeated one grep 322
  times): localizer thinking budget 1024 instead of off; "never repeat an identical tool
  call" rule in the shared prompt.

    uv run python -m src.build_submission phase_v2 <out_dir>
"""

from __future__ import annotations

import argparse
import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rbcore.paths import competition_dir

PARTS = Path(__file__).parent / "prompt_parts"
MODEL = "gemma-4-31b-it-qat-w4a16-ct"
GRAPH = ["search_similar_code", "get_code_neighbors", "get_code_subgraph"]
OUTPUT_KEY = {
    "localizer": "localization",
    "reproducer": "repro",
    "editor": "edit_summary",
    "verifier": "verification",
}


@dataclass(frozen=True)
class Variant:
    prompts: str
    """Folder under prompt_parts/."""
    pipeline: tuple[str, ...]
    """Phases run once, in order, before the fix loop (editor -> verifier)."""
    tools: dict[str, list[str]]
    thinking: dict[str, int] = field(default_factory=dict)
    """Per-phase thinking budget override: 0 disables thinking; absent = the base config."""
    max_fix_iterations: int = 3
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-phase sampling overrides (e.g. temperature) on top of that phase's thinking config."""


VARIANTS: dict[str, Variant] = {
    "phase_v1": Variant(
        prompts="v1",
        pipeline=("localizer", "reproducer"),
        max_fix_iterations=3,
        tools={
            "localizer": ["run_command", "read_file", "get_status", *GRAPH],
            "reproducer": ["run_command", "read_file", "get_status"],
            "editor": ["read_file", "edit_file", "write_file", "run_command", "get_status"],
            "verifier": ["run_command", "read_file", "get_status", "submit_patch"],
        },
        description="Locate -> reproduce -> (edit -> verify) x3, fresh context per phase.",
    ),
    "phase_v2": Variant(
        prompts="v2",
        pipeline=("localizer",),
        max_fix_iterations=2,
        thinking={"localizer": 0},
        tools={
            "localizer": ["run_command", "read_file", *GRAPH],
            "editor": ["read_file", "edit_file", "write_file", "run_command"],
            "verifier": ["run_command", "read_file", "submit_patch"],
        },
        description="Locate (no thinking, <=8 calls) -> (edit -> verify) x2, fresh contexts.",
    ),
    "phase_v2_1": Variant(
        prompts="v2_1",
        pipeline=("localizer",),
        max_fix_iterations=2,
        thinking={"localizer": 1024},
        tools={
            "localizer": ["run_command", "read_file", *GRAPH],
            "editor": ["read_file", "edit_file", "write_file", "run_command"],
            "verifier": ["run_command", "read_file", "submit_patch"],
        },
        description="Locate (thinking 1024, <=8 calls, no repeats) -> (edit -> verify) x2.",
    ),
    # v2.1 smoke: 2/10 (v2: 4/10); localizer thinking back on restored slow exploration
    # (2418 s of localizer model time vs 544 s in v2). v2.2 = v2 + the no-repeat rule only:
    # v2_1 prompts are exactly v2's prompts plus that rule.
    "phase_v2_2": Variant(
        prompts="v2_1",
        pipeline=("localizer",),
        max_fix_iterations=2,
        thinking={"localizer": 0},
        tools={
            "localizer": ["run_command", "read_file", *GRAPH],
            "editor": ["read_file", "edit_file", "write_file", "run_command"],
            "verifier": ["run_command", "read_file", "submit_patch"],
        },
        description="Locate (no thinking, <=8 calls, no repeats) -> (edit -> verify) x2.",
    ),
    # v2.2 G3 run (dev40 x 2): 28/79 task-runs looped (>= 10 identical tool calls, max 335:
    # the same empty grep), vs 1/80 for arm B; the prompt's no-repeat rule had no effect.
    # Loops sit in the thinking-off localizer (17) and the barely-thinking editor (11).
    # Submissions cannot add code or callbacks, and vLLM's frequency/presence penalties only
    # see the current completion, so the legal loop breaker is sampling: the starter's
    # temperature 0.2 is near-greedy; v2.3 uses Gemma's recommended 1.0 / 0.95 / top_k 64 in
    # the localizer and editor (the verifier, which submits, keeps 0.2).
    "phase_v2_3": Variant(
        prompts="v2_1",
        pipeline=("localizer",),
        max_fix_iterations=2,
        thinking={"localizer": 0},
        sampling={
            "localizer": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
            "editor": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        },
        tools={
            "localizer": ["run_command", "read_file", *GRAPH],
            "editor": ["read_file", "edit_file", "write_file", "run_command"],
            "verifier": ["run_command", "read_file", "submit_patch"],
        },
        description="v2.2 + Gemma-recommended sampling (T=1.0) in localizer and editor.",
    ),
    # Transfer-gap ladder (2026-09-27): on the scorer's engine (vLLM 0.19.1) thinking steps are
    # ~2x longer and the scorer ignores thinking_budget; thinking off + max_output 4096 (E1b
    # arm G) held 32.5% at L4-equivalent speed vs 26% for thinking capped at 4096 (arm H).
    # v2.4 = v2.3 with thinking off in every phase (max_output_tokens = reserve = 4096).
    "phase_v2_4": Variant(
        prompts="v2_1",
        pipeline=("localizer",),
        max_fix_iterations=2,
        thinking={"localizer": 0, "editor": 0, "verifier": 0},
        sampling={
            "localizer": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
            "editor": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        },
        tools={
            "localizer": ["run_command", "read_file", *GRAPH],
            "editor": ["read_file", "edit_file", "write_file", "run_command"],
            "verifier": ["run_command", "read_file", "submit_patch"],
        },
        description="v2.3 with thinking off in every phase (scorer-engine tuned).",
    ),
}


def phases(v: Variant) -> tuple[str, ...]:
    return (*v.pipeline, "editor", "verifier")


def phase_prompt(variant: str, phase: str) -> str:
    folder = PARTS / VARIANTS[variant].prompts
    return (folder / "shared.md").read_text() + (folder / f"{phase}.md").read_text()


def build(variant: str, out: Path, reserve: int = 4096, adapter: str | None = "main_lora") -> Path:
    v = VARIANTS[variant]
    if out.exists():
        shutil.rmtree(out)
    for sub in ("prompts", "sub_agents", "configs"):
        (out / sub).mkdir(parents=True)
    base = competition_dir() / "sample_submission"
    sampling = yaml.safe_load((base / "configs" / "sampling.yaml").read_text())
    thinking = int(sampling["thinking_config"]["thinking_budget"])
    sampling["max_output_tokens"] = thinking + reserve  # same reserve fix as E1b arm B
    configs: dict[str, dict[str, Any]] = {"sampling.yaml": sampling}
    for budget in sorted(set(v.thinking.values())):
        alt = copy.deepcopy(sampling)
        if budget == 0:
            alt["thinking_config"] = {"thinking_level": "NONE", "include_thoughts": False}
            alt["max_output_tokens"] = reserve
        else:
            alt["thinking_config"]["thinking_budget"] = budget
            alt["max_output_tokens"] = budget + reserve
        configs[f"sampling_think{budget}.yaml"] = alt
    phase_cfg = {
        p: f"sampling_think{v.thinking[p]}.yaml" if p in v.thinking else "sampling.yaml"
        for p in phases(v)
    }
    for phase, override in v.sampling.items():
        name = f"sampling_{phase}.yaml"
        configs[name] = {**configs[phase_cfg[phase]], **override}
        phase_cfg[phase] = name
    for name, body in configs.items():
        (out / "configs" / name).write_text(yaml.safe_dump(body, sort_keys=False))
    if adapter:
        shutil.copytree(base / "adapters", out / "adapters")

    for phase in phases(v):
        (out / "prompts" / f"{phase}.md").write_text(phase_prompt(variant, phase))
        agent = {
            "name": f"{phase}_agent",
            "model": MODEL,
            **({"adapter": adapter} if adapter else {}),
            "include_contents": "none",
            "output_key": OUTPUT_KEY[phase],
            "tools": v.tools[phase],
        }
        text = yaml.safe_dump(agent, sort_keys=False)
        text += (
            f"instruction: !include ../prompts/{phase}.md\n"
            f"generate_content_config: !include ../configs/{phase_cfg[phase]}\n"
        )
        (out / "sub_agents" / f"{phase}.yaml").write_text(text)

    (out / "sub_agents" / "fix_loop.yaml").write_text(
        yaml.safe_dump(
            {
                "agent_class": "LoopAgent",
                "name": "fix_loop",
                "max_iterations": v.max_fix_iterations,
                "sub_agents": [{"config_path": "editor.yaml"}, {"config_path": "verifier.yaml"}],
            },
            sort_keys=False,
        )
    )
    (out / "agent.yaml").write_text(
        yaml.safe_dump(
            {
                "agent_class": "SequentialAgent",
                "name": "phase_agent",
                "description": v.description,
                "sub_agents": [
                    *({"config_path": f"sub_agents/{p}.yaml"} for p in v.pipeline),
                    {"config_path": "sub_agents/fix_loop.yaml"},
                ],
            },
            sort_keys=False,
        )
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("out", type=Path)
    ap.add_argument("--reserve", type=int, default=4096)
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args()
    print(build(args.variant, args.out, args.reserve, None if args.no_adapter else "main_lora"))

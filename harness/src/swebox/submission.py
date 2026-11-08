"""Compile a declarative submission directory into an ADK agent tree (spec section 2-3).

Security rules from the spec are enforced: YAML only (``!include`` for .md/.txt/.yaml inside
the submission root, no absolute paths, ``..``, or symlinks escaping it), closed tool
registry, one base model across all agents.

Fidelity assumption A1 (documented in the E1 README): stock ADK ``LiteLlm`` drops
``thinking_config``. The organizers' ``adk-submission`` must translate it for vLLM; we map
it to the two request fields vLLM's Gemma 4 path understands:
``chat_template_kwargs.enable_thinking`` and ``thinking_token_budget``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from swebox.tools import ToolFn

MODEL_ALIASES = frozenset(
    {
        "gemma-4-31b-it-qat-w4a16-ct",
        "gemma-4-31b-it",
        "gemma-4-31b",
        "gemma-4-27b-it",
        "gemma-4-27b",
        "gemma-4-26b-a4b-it",
        "gemma-4-26b-a4b",
        "diffusiongemma-26b-a4b-it",
        "gemma-4-12b-it-qat-w4a16-ct",  # scorer models added 2026-11
        "gemma-4-e4b-it-qat-w4a16-ct",
        "gemma-4-e2b-it-qat-w4a16-ct",
        "gemma-4-12b-it",
        "gemma-4-12b",
        "gemma-4-9b-it",
        "gemma-4-9b",
        "gemma-4-e4b-it",
        "gemma-4-e4b",
        "gemma-4-e2b-it",
        "gemma-4-e2b",
    }
)
PROVIDER_PREFIXES = ("openai/", "google/", "hosted_vllm/", "custom/")
MAX_INCLUDE_DEPTH = 10
ROOT_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")


class SubmissionError(ValueError):
    pass


def normalize_model_name(name: str) -> str:
    for p in PROVIDER_PREFIXES:
        if name.startswith(p):
            name = name[len(p) :]
    return name


# -- YAML loading with sandboxed !include ----------------------------------------------------
class _Include:
    def __init__(self, rel: str) -> None:
        self.rel = rel


def _include_ctor(loader: yaml.SafeLoader, node: yaml.Node) -> _Include:
    return _Include(str(loader.construct_scalar(node)))  # type: ignore[arg-type]


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!include", _include_ctor)


def _safe_path(root: Path, base: Path, rel: str) -> Path:
    # ".." is allowed only while the result stays inside the root: the starter's sub-agent
    # includes ../prompts/analyzer.md. Absolute paths and null bytes are never allowed.
    if "\x00" in rel or Path(rel).is_absolute():
        raise SubmissionError(f"illegal include path {rel!r}")
    target = base / rel
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SubmissionError(f"include escapes submission root: {rel!r}")
    if target.is_symlink() and not target.resolve().is_relative_to(root.resolve()):
        raise SubmissionError(f"symlink escapes submission root: {rel!r}")
    return resolved


def load_yaml(root: Path, path: Path, depth: int = 0, seen: frozenset[Path] = frozenset()) -> Any:
    if depth > MAX_INCLUDE_DEPTH:
        raise SubmissionError("maximum !include depth exceeded")
    if path in seen:
        raise SubmissionError(f"!include cycle at {path}")
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    return _expand(root, path.parent, data, depth, seen | {path})


def _expand(root: Path, base: Path, node: Any, depth: int, seen: frozenset[Path]) -> Any:
    if isinstance(node, _Include):
        target = _safe_path(root, base, node.rel)
        if target.suffix in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        if target.suffix in (".yaml", ".yml"):
            return load_yaml(root, target, depth + 1, seen)
        raise SubmissionError(f"!include supports .md/.txt/.yaml only: {node.rel}")
    if isinstance(node, dict):
        return {k: _expand(root, base, v, depth, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(root, base, v, depth, seen) for v in node]
    return node


# -- compilation ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class ServingTarget:
    """How declared model aliases/adapters map onto the running vLLM server."""

    api_base: str = "http://127.0.0.1:8000/v1"
    served_model: str = "gemma-4-31b-it"
    num_retries: int = 5
    seed: int | None = None
    """If set, added to every agent's sampling config (reproducible multi-seed runs)."""
    thinking_translation: Literal["budget", "official"] = "budget"
    """How ``thinking_config`` reaches vLLM. "budget" (assumption A1, all runs before
    2026-09-27): enable_thinking + thinking_token_budget. "official": exactly what the scorer's
    adk_submission 0.2.11 sends (resolvers/generation.py): enable_thinking + reasoning_effort and
    NO budget, so thinking is bounded only by max_output_tokens."""


@dataclass
class Compiled:
    root: BaseAgent
    model: str
    adapters: list[str]
    agent_count: int


def find_root(submission: Path) -> Path:
    found = [submission / n for n in ROOT_NAMES if (submission / n).is_file()]
    if not found:
        raise SubmissionError("MissingRootConfigError: no agent.yaml / root_agent.yaml")
    if len(found) > 1:
        raise SubmissionError(f"MultipleRootConfigsError: {[p.name for p in found]}")
    return found[0]


def compile_submission(
    submission: Path, tools: Mapping[str, ToolFn], target: ServingTarget
) -> Compiled:
    """Build a fresh agent tree (call once per task: tools are bound to that task)."""
    submission = submission.resolve()
    state = _State(submission, tools, target)
    root = state.agent(load_yaml(submission, find_root(submission)), submission)
    if len(state.models) > 1:
        raise SubmissionError(f"single base model rule violated: {sorted(state.models)}")
    return Compiled(
        root, next(iter(state.models), target.served_model), sorted(state.adapters), state.count
    )


class _State:
    def __init__(self, root: Path, tools: Mapping[str, ToolFn], target: ServingTarget) -> None:
        self.root, self.tools, self.target = root, tools, target
        self.models: set[str] = set()
        self.adapters: set[str] = set()
        self.count = 0

    def agent(self, cfg: dict[str, Any], base: Path) -> BaseAgent:
        self.count += 1
        cls = cfg.get("agent_class", "LlmAgent")
        common = {"name": cfg["name"], "description": cfg.get("description", "")}
        subs = [self.ref(r, base) for r in cfg.get("sub_agents", [])]
        if cls == "SequentialAgent":
            return SequentialAgent(**common, sub_agents=subs)
        if cls == "ParallelAgent":
            return ParallelAgent(**common, sub_agents=subs)
        if cls == "LoopAgent":
            n = int(cfg.get("max_iterations", 500))
            if not 1 <= n <= 500:
                raise SubmissionError("LoopAgent.max_iterations must be in 1..500")
            return LoopAgent(**common, sub_agents=subs, max_iterations=n)
        if cls != "LlmAgent":
            raise SubmissionError(f"unsupported agent_class {cls!r}")
        if cfg.get("skills"):
            raise SubmissionError("skills are not supported by swebox yet")

        model = normalize_model_name(cfg["model"])
        if model not in MODEL_ALIASES:
            raise SubmissionError(f"unknown model alias {model!r}")
        self.models.add(model)
        adapter = cfg.get("adapter")
        if adapter:
            if not (self.root / "adapters" / adapter / "adapter_model.safetensors").exists():
                raise SubmissionError(f"adapter {adapter!r} not found under adapters/")
            self.adapters.add(adapter)
        served = adapter or self.target.served_model
        llm = LiteLlm(
            model=f"openai/{served}",
            api_base=self.target.api_base,
            api_key="EMPTY",
            num_retries=self.target.num_retries,
        )

        kwargs: dict[str, Any] = {
            **common,
            "model": llm,
            "instruction": cfg.get("instruction", ""),
            "tools": [self.tool(t, base) for t in cfg.get("tools", [])],
            "sub_agents": subs,
            "generate_content_config": self.gen_config(cfg.get("generate_content_config") or {}),
        }
        for key in (
            "output_key",
            "include_contents",
            "disallow_transfer_to_parent",
            "disallow_transfer_to_peers",
        ):
            if key in cfg:
                kwargs[key] = cfg[key]
        return LlmAgent(**kwargs)

    def ref(self, ref: dict[str, Any], base: Path) -> BaseAgent:
        if "config_path" in ref:
            path = _safe_path(self.root, base, ref["config_path"])
            return self.agent(load_yaml(self.root, path), path.parent)
        return self.agent(ref, base)  # inline agent dict

    def tool(self, spec: str | dict[str, Any], base: Path) -> Any:
        if isinstance(spec, str):
            if spec not in self.tools:
                raise SubmissionError(f"unknown or unavailable tool {spec!r}")
            return FunctionTool(self.tools[spec])
        if "agent_tool" in spec:
            at = spec["agent_tool"]
            return AgentTool(
                self.ref(at, base), skip_summarization=bool(at.get("skip_summarization"))
            )
        return AgentTool(self.agent(spec, base))

    def gen_config(self, raw: dict[str, Any]) -> types.GenerateContentConfig:
        raw = dict(raw)
        for banned in (
            "tools",
            "system_instruction",
            "http_options",
            "safety_settings",
            "response_schema",
        ):
            if banned in raw:
                raise SubmissionError(f"generate_content_config.{banned} is not allowed")
        thinking = raw.pop("thinking_config", None) or {}
        if self.target.seed is not None:
            raw.setdefault("seed", self.target.seed)
        max_out = int(raw.setdefault("max_output_tokens", 16_384))
        if not 1 <= max_out <= 32_768:
            raise SubmissionError("max_output_tokens must be in 1..32768")
        if self.target.thinking_translation == "official":
            thinking_extra_body(thinking)  # same validation of the declared budget
            extra = official_thinking_body(thinking)
        else:
            extra = thinking_extra_body(thinking)
        return types.GenerateContentConfig(
            **raw, http_options=types.HttpOptions(extra_body=extra) if extra else None
        )


def thinking_extra_body(thinking: dict[str, Any]) -> dict[str, Any]:
    """Fidelity assumption A1: map ADK thinking_config to vLLM request fields."""
    if not thinking:
        return {}
    level = str(thinking.get("thinking_level", "")).upper()
    enabled = level != "NONE" and thinking.get("include_thoughts", True) is not False
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": enabled}}
    if enabled:
        budget = int(thinking.get("thinking_budget", 4096))
        if not 1 <= budget <= 32_768:
            raise SubmissionError("thinking_budget must be in 1..32768")
        body["thinking_token_budget"] = budget
    return body


THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def official_thinking_body(thinking: dict[str, Any]) -> dict[str, Any]:
    """The scorer's mapping (adk_submission 0.2.11 apply_thinking_config_to_model, wheel of
    2026-09-27), in request fields. thinking_budget only switches thinking (<= 0 off, > 0 on);
    it is never sent as a token limit. The scorer's server defaults enable_thinking to True
    (getting-started notebook), which is made explicit here when the config leaves it open."""
    include = thinking.get("include_thoughts")
    raw = thinking.get("thinking_level")
    level = str(getattr(raw, "value", raw)).lower() if raw is not None else None
    budget = thinking.get("thinking_budget")
    if include is False or level == "none" or (budget is not None and int(budget) <= 0):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": True}}
    if (include is True or level in THINKING_LEVELS) and level in {"low", "medium", "high"}:
        body["reasoning_effort"] = level
    return body

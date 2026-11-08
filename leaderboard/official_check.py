"""Validate and compile leaderboard builds with the scorer's own code (official wheelhouse:
adk_submission 0.2.11, swegemma 0.2.7, google-adk 1.36.1), so a packaging mistake cannot waste a
daily submission. No model is called; tools are stand-ins with the real signatures.

    infra/official/.venv/bin/python leaderboard/official_check.py leaderboard/lb03_thinking_off ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from adk_submission import ModelRegistry, compile_submission, validate_directory
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import build_submission_limits

try:  # swegemma <= 0.2.7; removed in 0.2.11 (the compiler enforces the single-model rule)
    from swegemma.models.discovery import validate_single_declared_model
except ImportError:
    validate_single_declared_model = None

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
# the scorer's model choices (one per submission; added 2026-11)
MODELS = (
    MODEL,
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
)


def run_command(command: str) -> str: ...
def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str: ...
def write_file(filepath: str, content: str) -> str: ...
def edit_file(
    filepath: str, old_string: str, new_string: str, allow_multiple: bool = False
) -> str: ...
def submit_patch() -> str: ...
def get_status() -> str: ...
def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str: ...
def search_similar_code(query: str, k: int = 10) -> str: ...
def get_code_subgraph(nodes: list[str]) -> str: ...


TOOLS = {
    f.__name__: f
    for f in (
        run_command,
        read_file,
        write_file,
        edit_file,
        submit_patch,
        get_status,
        get_code_neighbors,
        search_similar_code,
        get_code_subgraph,
    )
}


def check(bundle: Path) -> str:
    limits, constraints = build_submission_limits()
    registry = ModelRegistry()
    for model in MODELS:
        registry.register(
            model,
            LiteLlm(model=f"openai/{model}", api_base="http://127.0.0.1:9/v1", api_key="EMPTY"),
        )
    validate_directory(bundle, limits)
    if validate_single_declared_model is not None:
        assert validate_single_declared_model(bundle) in MODELS
    root = compile_submission(
        bundle, TOOLS, registry, limits=limits, generation_constraints=constraints
    )
    return f"ok (root agent {root.name!r})"


if __name__ == "__main__":
    failed = 0
    for arg in sys.argv[1:]:
        try:
            print(f"{arg}: {check(Path(arg))}")
        except Exception as error:  # report every bundle, then fail
            failed += 1
            print(f"{arg}: FAILED {type(error).__name__}: {str(error)[:400]}")
    sys.exit(1 if failed else 0)

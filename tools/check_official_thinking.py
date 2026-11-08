"""Cross-check swebox's official thinking translation against the scorer's own code
(adk_submission.resolvers.generation.apply_thinking_config_to_model) for a grid of configs.

    infra/official/.venv/bin/python tools/check_official_thinking.py
Exits non-zero on any mismatch. Run after every wheelhouse update.
"""

from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from adk_submission.resolvers.generation import apply_thinking_config_to_model

# Load only official_thinking_body (+ THINKING_LEVELS) from swebox: this venv holds the scorer's
# packages, not swebox's dependencies.
SRC = Path(__file__).resolve().parents[1] / "harness" / "src" / "swebox" / "submission.py"
tree = ast.parse(SRC.read_text())
keep = [
    n
    for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name == "official_thinking_body")
    or (
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "THINKING_LEVELS" for t in n.targets)
    )
]
ns: dict[str, Any] = {"Any": Any}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
official_thinking_body = ns["official_thinking_body"]

mismatches = 0
for include, level, budget in itertools.product(
    [None, True, False], [None, "none", "minimal", "low", "high"], [None, 0, 4096]
):
    cfg = {
        k: v
        for k, v in (
            ("include_thoughts", include),
            ("thinking_level", level),
            ("thinking_budget", budget),
        )
        if v is not None
    }
    model = SimpleNamespace(_additional_args={})
    out = apply_thinking_config_to_model(model, {"thinking_config": cfg} if cfg else {})
    args = getattr(out, "_additional_args", {})
    official = {}
    if "extra_body" in args:
        official["chat_template_kwargs"] = args["extra_body"]["chat_template_kwargs"]
    if "reasoning_effort" in args:
        official["reasoning_effort"] = args["reasoning_effort"]
    if not official:  # official leaves the request unchanged: the server default (thinking on)
        official = {"chat_template_kwargs": {"enable_thinking": True}}
    ours = official_thinking_body(cfg)
    if ours != official:
        mismatches += 1
        print("MISMATCH", cfg, "official:", official, "swebox:", ours)
print(f"{mismatches} mismatches")
sys.exit(1 if mismatches else 0)

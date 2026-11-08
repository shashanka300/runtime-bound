"""Ablation arms built from the starter submission. Each arm changes one thing relative to its
parent, so paired differences isolate each fix.

A  starter as-is
B  A + ``max_output_tokens = thinking_budget + output_reserve`` (starter: 16384, which leaves a
   16k usable prompt because vLLM rejects prompt + max_tokens > 32768)
C  B + GUIDANCE appended to the system prompt (small targeted reads; scratch via run_command)
D  B + fresh-context retry after a context death, same workspace (harness-level diagnostic,
   ``Budget.context_retries``; not implementable inside an official submission)
E  B + Gemma's recommended sampling (temperature 1.0, top_p 0.95, top_k 64); added after E5
   showed the starter's 0.2 drives repeated-call loops in low-thinking phases. Controls whether
   E5 phase_v2_3's gain comes from sampling rather than the phase architecture.
F  B without the starter's LoRA adapters (base model): step 1 of the transfer-gap ladder.
   Kaggle's scorer (patched vLLM 0.19.1) silently wipes LoRA weights, so B scores as F there.
G  F with thinking off and ``max_output_tokens`` = output_reserve (4096). The scorer never sends
   a thinking budget, so thinking is bounded only by max_output_tokens; switching it off is the
   choice of the best public notebooks. Meant for thinking_translation="official".
H  F with thinking on but ``max_output_tokens`` = output_reserve (4096): on the scorer this is
   the only cap on thinking that takes effect.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

GUIDANCE = """

## Context budget (important)
Your context window is small and every tool output stays in it until the task ends.
- Locate code before reading it: use `grep -n` (or `search_similar_code`) to find the exact
  lines, then `read_file` with `start_line`/`end_line` spanning at most ~80 lines around them.
  Never read a whole large file.
- `write_file` always writes inside /workspace. Create scratch or reproduction scripts only
  through `run_command`, e.g. `cat > /tmp/repro.py <<'EOF' ... EOF`, so they stay out of the
  patch.
- Keep your reasoning short; move to an edit as soon as you know where the bug is.
"""

ARMS = ("A", "B", "C", "D", "E", "F", "G", "H")
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def make_arm(base: Path, out: Path, arm: str, output_reserve: int) -> Path:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    if arm == "A":
        return out
    sampling = out / "configs" / "sampling.yaml"
    cfg = yaml.safe_load(sampling.read_text()) or {}
    thinking = int((cfg.get("thinking_config") or {}).get("thinking_budget", 4096))
    cfg["max_output_tokens"] = thinking + output_reserve
    if arm == "E":
        cfg.update(GEMMA_SAMPLING)
    if arm in ("G", "H"):
        cfg["max_output_tokens"] = output_reserve
    if arm == "G":
        cfg["thinking_config"] = {"include_thoughts": False}
    sampling.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if arm == "C":
        system = out / "prompts" / "system.md"
        system.write_text(system.read_text() + GUIDANCE)
    if arm in ("F", "G", "H"):
        drop_adapters(out)
    return out


def drop_adapters(out: Path) -> None:
    """Remove adapters/ and every ``adapter:`` line (YAML with !include tags: edit as text)."""
    shutil.rmtree(out / "adapters", ignore_errors=True)
    for y in [*out.rglob("*.yaml"), *out.rglob("*.yml")]:
        lines = y.read_text().splitlines(keepends=True)
        kept = [ln for ln in lines if not ln.lstrip().startswith("adapter:")]
        if kept != lines:
            y.write_text("".join(kept))


def context_retries(arm: str, retries: int) -> int:
    return retries if arm == "D" else 0

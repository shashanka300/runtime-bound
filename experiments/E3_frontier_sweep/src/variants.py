"""Submission variants for the sweep: same agent, different base model / thinking budget.

The starter submission is copied and changed in exactly three places:
- every ``model:`` line -> the condition's model alias (single-model rule preserved);
- ``adapter:`` lines removed and ``adapters/`` not copied (the starter's rank-4 LoRAs
  target the 31B and cannot load on 12B; both models run adapter-free for parity);
- ``configs/sampling.yaml``: ``thinking_budget`` set to the condition and
  ``max_output_tokens = thinking_budget + output_reserve`` (the starter's fixed 16384
  reserve halves the usable prompt, see E2, and would mask the model/thinking effect).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

_MODEL = re.compile(r"^(\s*)model:\s*\S+\s*$", re.MULTILINE)
_ADAPTER = re.compile(r"^\s*adapter:\s*\S+\s*\n", re.MULTILINE)


def make_variant(
    base: Path, out: Path, model_alias: str, thinking_budget: int, output_reserve: int
) -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("adapters"))
    for f in [*out.glob("*.yaml"), *out.glob("*.yml"), *out.glob("sub_agents/*.y*ml")]:
        text = _ADAPTER.sub("", f.read_text())
        f.write_text(_MODEL.sub(lambda m: f"{m.group(1)}model: {model_alias}", text))
    sampling = out / "configs" / "sampling.yaml"
    cfg = yaml.safe_load(sampling.read_text()) or {}
    cfg.setdefault("thinking_config", {})["thinking_budget"] = thinking_budget
    cfg["max_output_tokens"] = thinking_budget + output_reserve
    sampling.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out

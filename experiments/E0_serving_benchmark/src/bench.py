"""Serving benchmark cells, driven through vLLM's own ``vllm bench serve``.

Using the upstream benchmark client (rather than a home-grown one) keeps the
throughput/latency numbers comparable with published vLLM results.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from rbcore.config import ServingConfig
from rbcore.paths import data_root, repo_root

from .params import E0Params

# Headroom so input + output never exceeds max_model_len after chat/special tokens.
SAFETY_TOKENS = 64
VLLM_BIN = repo_root() / "infra/vllm/.venv/bin/vllm"


@dataclass(frozen=True)
class Cell:
    prefix_len: int
    context_len: int
    concurrency: int
    output_len: int

    @property
    def input_len(self) -> int:
        """Random (non-shared) prompt tokens; prefix + input + output fills the context."""
        return self.context_len - self.output_len - self.prefix_len - SAFETY_TOKENS

    @property
    def key(self) -> str:
        return f"p{self.prefix_len}_c{self.context_len}_n{self.concurrency}"


def grid(p: E0Params) -> list[Cell]:
    cells = [
        Cell(pre, ctx, conc, p.output_len)
        for pre in p.prefix_lens
        for ctx in p.context_lens
        for conc in p.concurrency
    ]
    return [c for c in cells if c.input_len > 0]


def num_prompts(cell: Cell, p: E0Params) -> int:
    return max(p.min_prompts, cell.concurrency * p.prompts_per_slot)


def bench_cmd(
    cell: Cell, p: E0Params, serving: ServingConfig, out_dir: Path, seed: int, n: int
) -> list[str]:
    return [
        str(VLLM_BIN), "bench", "serve",
        "--base-url", serving.base_url,
        "--model", serving.model,
        "--tokenizer", p.tokenizer,
        "--dataset-name", "random",
        "--random-input-len", str(cell.input_len),
        "--random-output-len", str(cell.output_len),
        "--random-prefix-len", str(cell.prefix_len),
        "--random-range-ratio", "0",
        "--num-prompts", str(n),
        "--max-concurrency", str(cell.concurrency),
        "--ignore-eos",
        "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result", "--save-detailed",
        "--result-dir", str(out_dir),
        "--result-filename", f"{cell.key}.json",
    ]  # fmt: skip


def run_cell(cmd: list[str], log_path: Path) -> None:
    """Run one benchmark subprocess; its stdout/stderr go to a per-cell log file."""
    env = os.environ | {"HF_HOME": str(data_root() / "hf")}
    with log_path.open("w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=True)


SCALAR_KEYS = re.compile(
    r"^(completed|duration|total_input_tokens|total_output_tokens|request_throughput|"
    r"output_throughput|total_token_throughput|(mean|median|std|p\d+)_(ttft|tpot|itl|e2el)_ms)$"
)


def load_results(result_dir: Path, cells: list[Cell]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in cells:
        path = result_dir / f"{c.key}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        rows.append(
            {
                **asdict(c),
                "input_len": c.input_len,
                "key": c.key,
                **{k: v for k, v in res.items() if SCALAR_KEYS.match(k)},
            }
        )
    return pl.DataFrame(rows)


def smoke_test(serving: ServingConfig) -> dict[str, Any]:
    """Correctness probe: broken quantized kernels usually produce fluent-looking garbage."""
    r = (
        httpx.post(
            f"{serving.base_url}/v1/chat/completions",
            json={
                "model": serving.model,
                "messages": [
                    {"role": "user", "content": "What is 17 * 23? Reply with only the number."}
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            },
            timeout=serving.request_timeout_s,
        )
        .raise_for_status()
        .json()
    )
    content = r["choices"][0]["message"].get("content") or ""
    return {"content": content.strip(), "ok": "391" in content, "usage": r.get("usage")}


_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_MAX_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def kv_capacity(log_file: Path) -> dict[str, float | None]:
    """KV-cache capacity as reported by vLLM at startup (last server start in the log)."""
    text = log_file.read_text(errors="replace") if log_file.exists() else ""
    kv = _KV_TOKENS.findall(text)
    conc = _MAX_CONC.findall(text)
    return {
        "kv_cache_tokens": float(kv[-1].replace(",", "")) if kv else None,
        "max_concurrency_at_len": float(conc[-1][1]) if conc else None,
        "at_len": float(conc[-1][0].replace(",", "")) if conc else None,
    }

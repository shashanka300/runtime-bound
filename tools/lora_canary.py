"""LoRA canary: does a served engine actually apply adapter weights?

`make`: copy the starter main_lora (rank 4, layer-0 q_proj/o_proj, lora_B all zero, i.e. a no-op)
and fill every lora_B with N(0, 1) values, so an applied adapter must change the output.
`compare`: send one fixed prompt at temperature 0 to the base model, main_lora and the canary,
and report whether the outputs and top-5 logprobs differ.

    infra/vllm019/.venv/bin/python tools/lora_canary.py make
    infra/vllm019/.venv/bin/python tools/lora_canary.py compare [label]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "competition" / "sample_submission" / "adapters" / "main_lora"
CANARY = ROOT / "data" / "probes" / "lora_canary"


def make() -> None:
    import torch
    from safetensors.torch import load_file, save_file

    if CANARY.exists():
        shutil.rmtree(CANARY)
    shutil.copytree(STARTER, CANARY)
    path = CANARY / "adapter_model.safetensors"
    tensors = load_file(str(path))
    scale = float(os.environ.get("CANARY_SCALE", "1"))  # 50 makes an applied adapter obvious
    gen = torch.Generator().manual_seed(0)
    for k, t in tensors.items():
        if "lora_B" in k:
            noise = torch.randn(t.shape, generator=gen, dtype=torch.float32) * scale
            tensors[k] = noise.to(t.dtype)
    save_file(tensors, str(path))
    print(CANARY, {k: float(v.abs().max()) for k, v in tensors.items() if "lora_B" in k})


def compare(label: str) -> None:
    import httpx

    prompt = "Write one sentence about the ocean."
    out = {}
    for model in ("gemma-4-31b-it", "main_lora", "canary"):
        r = httpx.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 24,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=300,
        ).json()
        choice = r["choices"][0]
        first = choice["logprobs"]["content"][0]["top_logprobs"]
        out[model] = {
            "text": choice["message"]["content"],
            "first_token_top5": [(t["token"], round(t["logprob"], 4)) for t in first],
        }
    base = out["gemma-4-31b-it"]
    result = {
        "label": label,
        "main_lora_equals_base": out["main_lora"] == base,
        "canary_equals_base": out["canary"] == base,
        "outputs": out,
    }
    dest = ROOT / "data" / "probes" / f"lora_canary_{label}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        compare(sys.argv[2] if len(sys.argv) > 2 else "run")

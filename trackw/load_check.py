"""Track W3 step 0: load Gemma 4 12B (QAT int4 checkpoint, decompressed to bf16) in
transformers -- the model the editor's forward passes and activation reads will use -- and check
it matches the vLLM 0.19.1 server (same checkpoint, host hf-overrides) on greedy decoding.

    infra/vllm019/.venv/bin/python trackw/load_check.py [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SNAP = next((ROOT / "data/hf/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots").iterdir())
PROMPT = [
    {
        "role": "user",
        "content": "Write a Python function that returns the n-th Fibonacci "
        "number. Answer with code only.",
    }
]


def load(device: str):
    tok = AutoTokenizer.from_pretrained(SNAP)
    t0 = time.time()
    # no device_map: it needs accelerate, which the scorer-matched env does not have
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s on {device}")
    return tok, model


def vllm_greedy(n: int) -> str:
    body = {
        "model": "gemma-4-12b-it-qat-w4a16-ct",
        "messages": PROMPT,
        "max_tokens": n,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"][
        "content"
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", type=int, default=24)
    args = ap.parse_args()
    tok, model = load(args.device)
    ids = tok.apply_chat_template(
        PROMPT,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
        return_dict=True,
    )
    ids = {k: v.to(model.device) for k, v in ids.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.tokens, do_sample=False)
    hf = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"hf greedy ({time.time() - t0:.0f}s): {hf!r}")
    vl = vllm_greedy(args.tokens)
    print(f"vllm greedy: {vl!r}")
    common = 0
    for a, b in zip(hf, vl, strict=False):
        if a != b:
            break
        common += 1
    print(
        f"common prefix {common} chars of {min(len(hf), len(vl))}: "
        f"{'MATCH' if common >= min(len(hf), len(vl)) - 2 else 'DIFFER'}"
    )


if __name__ == "__main__":
    main()

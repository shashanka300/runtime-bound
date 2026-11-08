"""Track W4 canary v2: is our 12B LoRA applied by vLLM, and does it equal the merged delta?
Compares next-token top-20 logprobs (base vs adapter on vLLM; adapter vs transformers with the
delta hooked in) on ordinary prompts, instead of greedy text (v1's prompt degenerated to "1111").

    infra/vllm019/.venv/bin/python trackw/lora_verify2.py data/trackw/canary12b
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts  # noqa: E402

PROMPTS = ["The capital of France is", "import numpy as np\nx = np.", "Once upon a time, there"]


def top(model: str, prompt: str) -> dict[str, float]:
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return out["choices"][0]["logprobs"]["top_logprobs"][0]


def shift(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    return max((abs(a[k] - b[k]) for k in keys), default=float("inf"))


def main() -> None:
    adapter = Path(sys.argv[1])
    base = {p: top("gemma-4-12b-it-qat-w4a16-ct", p) for p in PROMPTS}
    lora = {p: top("canary", p) for p in PROMPTS}
    for p in PROMPTS:
        print(f"{p!r}: base vs lora max |dlogprob| {shift(base[p], lora[p]):.3f}; "
              f"top base {max(base[p], key=base[p].get)!r} lora {max(lora[p], key=lora[p].get)!r}")
    from transformers import AutoTokenizer

    model = acts.load("cpu")
    blocks = acts.decoder_layers(model)
    for name, d in torch.load(adapter.with_suffix(".delta.pt")).items():
        li, mod = int(name.split(".")[1]), name.split(".", 2)[2]
        dd = d.to(torch.bfloat16)
        blocks[li].get_submodule(mod).register_forward_hook(
            lambda _m, inp, out, dd=dd: out + torch.nn.functional.linear(inp[0], dd)
        )
    tok = AutoTokenizer.from_pretrained(acts.SNAP)
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")
        with torch.no_grad():
            lp = torch.log_softmax(model(**ids).logits[0, -1].float(), -1)
        merged = {tok.decode([int(t)]): float(lp[t]) for t in lp.topk(20).indices}
        print(f"{p!r}: merged vs lora max |dlogprob| {shift(merged, lora[p]):.3f} "
              f"(vs base {shift(merged, base[p]):.3f})")


if __name__ == "__main__":
    main()

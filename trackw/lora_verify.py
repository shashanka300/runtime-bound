"""Track W4: does the scorer's vLLM apply our 12B LoRA exactly, and what does it cost?

Needs vLLM 0.19.1 serving 12B with --enable-lora --lora-modules canary=<adapter dir>.
  1. greedy 48 tokens from base and from the adapter: must differ (adapter applied);
  2. the same delta merged into transformers weights (CPU): adapter output must match it;
  3. decode tokens/s with and without the adapter (the forum suspects LoRA slows decoding).

    infra/vllm019/.venv/bin/python trackw/lora_verify.py data/trackw/canary12b
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts

PROMPT = "def quicksort(arr):\n    "
MODEL = "gemma-4-12b-it-qat-w4a16-ct"


def complete(model: str, n: int) -> tuple[str, float]:
    body = {"model": model, "prompt": PROMPT, "max_tokens": n, "temperature": 0}
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return out["choices"][0]["text"], out["usage"]["completion_tokens"] / (time.time() - t0)


def main() -> None:
    adapter = Path(sys.argv[1])
    base, _ = complete(MODEL, 48)
    lora, _ = complete("canary", 48)
    _, tps_base = complete(MODEL, 512)
    _, tps_lora = complete("canary", 512)
    print(f"base : {base!r}\nlora : {lora!r}")
    print(
        f"applied: {base != lora}; decode tok/s base {tps_base:.1f}, lora {tps_lora:.1f} "
        f"({100 * (tps_lora / tps_base - 1):+.1f}%)",
        flush=True,
    )
    deltas = torch.load(adapter.with_suffix(".delta.pt"))
    model = acts.load("cpu")
    blocks = acts.decoder_layers(model)
    # linears stay int4-packed in transformers (dequantised inside forward): apply the edit as
    # an output hook, out += x @ delta^T, which is exactly what the LoRA computes
    for name, d in deltas.items():
        li, mod = int(name.split(".")[1]), name.split(".", 2)[2]
        lin = blocks[li].get_submodule(mod)
        dd = d.to(torch.bfloat16)
        lin.register_forward_hook(
            lambda _m, inp, out, dd=dd: out + torch.nn.functional.linear(inp[0], dd)
        )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(acts.SNAP)
    ids = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=48, do_sample=False)
    merged = tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
    common = next(
        (k for k, (a, b) in enumerate(zip(merged, lora, strict=False)) if a != b),
        min(len(merged), len(lora)),
    )
    print(
        f"merged: {merged!r}\nadapter vs merged common prefix {common}/{min(len(merged), len(lora))}"
        f": {'MATCH' if common >= min(len(merged), len(lora)) - 2 else 'DIFFER'}"
    )


if __name__ == "__main__":
    main()

"""Track W3: residual-stream activations of Gemma 4 12B (transformers, bf16 decompressed from the
QAT int4 checkpoint; verified to match vLLM greedily by trackw/load_check.py) at every decision
point, with ONE forward pass per task: within a task each request's prompt is a prefix of the
next, so the last request's prompt contains every earlier decision position.

    infra/vllm019/.venv/bin/python trackw/acts.py data/screen_nov/lb06_12b_requests.jsonl \
        --layers 8,16,24,32,40,47 --device cuda
writes <records>.acts.pt: {"i": [...], "layers": [...], "h": tensor[n_points, n_layers, d]} with
h = residual stream after each listed decoder layer at the last prompt token of each request.
Points whose prompt is not a prefix of the task's final prompt (context compaction) get their own
forward pass.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

CHUNK = 4096
ROOT = Path(__file__).resolve().parents[1]
SNAP = next((ROOT / "data/hf/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots").iterdir())


def load(device: str):
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16).to(device)
    model.eval()
    return model


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("language_model.layers") or name.endswith("model.layers"):
            return mod
    raise RuntimeError("decoder layers not found")


@torch.no_grad()
def run(model, ids: list[int], positions: list[int], layers: list[int]) -> torch.Tensor:
    """At `positions`, for each layer in `layers`: [0] the residual stream after the layer and
    [1] the MLP input (after pre_feedforward_layernorm) -> [len(pos), len(layers), 2, d]."""
    from transformers import DynamicCache

    store: dict[tuple[int, int], list[torch.Tensor]] = {
        (li, c): [] for li in layers for c in (0, 1)
    }
    hooks = []
    blocks = decoder_layers(model)
    window = {"lo": 0, "hi": 0}
    wanted = sorted(set(positions))

    def grab(key, hs):
        sel = [p - window["lo"] for p in wanted if window["lo"] <= p < window["hi"]]
        if sel:
            store[key].append(hs[0, sel].float().cpu())

    for li in layers:

        def hook(_m, _inp, out, li=li):
            grab((li, 0), out[0] if isinstance(out, tuple) else out)

        def mlp_hook(_m, inp, li=li):
            grab((li, 1), inp[0])

        hooks.append(blocks[li].register_forward_hook(hook))
        hooks.append(blocks[li].mlp.register_forward_pre_hook(mlp_hook))
    # chunked prefill with a KV cache: attention memory stays bounded for 30k-token prompts
    cache = DynamicCache(config=model.config)
    try:
        for lo in range(0, len(ids), CHUNK):
            hi = min(lo + CHUNK, len(ids))
            window.update(lo=lo, hi=hi)
            model(
                input_ids=torch.tensor([ids[lo:hi]], device=model.device),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(lo, hi, device=model.device),
                logits_to_keep=1,
            )
    finally:
        for h in hooks:
            h.remove()
    order = {p: n for n, p in enumerate(wanted)}
    h = torch.stack(
        [torch.stack([torch.cat(store[(li, c)]) for c in (0, 1)], dim=1) for li in layers], dim=1
    )
    return h[[order[p] for p in positions]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", type=Path)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    layers = list(range(48)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    toks = {}
    for ln in args.records.with_suffix(".tokens.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["ok"]:
            toks[r["i"]] = r["prompt_ids"]
    points = [
        json.loads(ln) for ln in args.records.with_suffix(".points.jsonl").read_text().splitlines()
    ]
    by_task = defaultdict(list)
    for p in points:
        if p["i"] in toks:
            by_task[p["task"]].append(p["i"])
    model = load(args.device)
    out_i, out_h = [], []
    t0 = time.time()
    for n, (task, idx) in enumerate(by_task.items(), 1):
        idx.sort()
        pending = list(idx)
        while pending:
            full = toks[pending[-1]]
            covered = [i for i in pending if toks[i] == full[: len(toks[i])]]
            h = run(model, full, [len(toks[i]) - 1 for i in covered], layers)
            out_i += covered
            out_h.append(h.to(torch.bfloat16))  # bf16: Gemma residuals can exceed fp16 range
            pending = [i for i in pending if i not in covered]
        print(
            f"[{n}/{len(by_task)}] {task[:40]!r}: {len(idx)} points, {time.time() - t0:.0f}s",
            flush=True,
        )
    dst = args.records.with_suffix(".acts.pt")
    torch.save({"i": out_i, "layers": layers, "h": torch.cat(out_h)}, dst)
    print(dst)


if __name__ == "__main__":
    main()

"""Track W4: package a weight edit as a PEFT LoRA adapter the scorer's vLLM can mount.

An edit is a dict {"layers.<N>.mlp.down_proj": delta_W [out, in], ...} of additive changes to the
bf16 *dequantised* QAT weights (what the scorer computes with). Each delta is factorised by SVD to
rank r (vLLM ranks: 1, 8, 16, 32, ...), with lora_alpha = r so the scale is 1:
    W + B @ A,  B = U_r * S_r  [out, r],  A = V_r^T [r, in]

    python trackw/lora_export.py canary --out data/trackw/canary12b --layer 20 --scale 0.05
    python trackw/lora_export.py export edit.pt --out data/trackw/edit_v1 --rank 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

BASE = "google/gemma-4-12b-it-qat-w4a16-ct"
PREFIX = "base_model.model.model.language_model."
SHAPES = {  # Gemma 4 12B text decoder
    "mlp.gate_proj": (15360, 3840),
    "mlp.up_proj": (15360, 3840),
    "mlp.down_proj": (3840, 15360),
}


def factorise(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    b = u[:, :rank] * s[:rank]
    a = vh[:rank]
    kept = float((s[:rank] ** 2).sum() / (s**2).sum().clamp_min(1e-30))
    return a.contiguous(), b.contiguous(), kept


def write_adapter(deltas: dict[str, torch.Tensor], out: Path, rank: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tensors, report, modules, layers = {}, {}, set(), set()
    for name, delta in deltas.items():
        a, b, kept = factorise(delta, rank)
        tensors[f"{PREFIX}{name}.lora_A.weight"] = a.to(torch.bfloat16)
        tensors[f"{PREFIX}{name}.lora_B.weight"] = b.to(torch.bfloat16)
        report[name] = {"energy_kept": round(kept, 6), "delta_norm": float(delta.norm())}
        modules.add(name.split(".")[-1])
        layers.add(int(name.split(".")[1]))
    save_file(tensors, out / "adapter_model.safetensors")
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted(modules),
        "layers_to_transform": sorted(layers),
        "use_dora": False,
        "use_rslora": False,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canary")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--layer", type=int, default=20)
    c.add_argument("--scale", type=float, default=0.05)
    c.add_argument("--rank", type=int, default=8)
    e = sub.add_parser("export")
    e.add_argument("edit", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "canary":
        g = torch.Generator().manual_seed(0)
        o, i = SHAPES["mlp.down_proj"]
        delta = torch.randn(o, args.rank, generator=g) @ torch.randn(args.rank, i, generator=g)
        delta *= args.scale / args.rank**0.5
        deltas = {f"layers.{args.layer}.mlp.down_proj": delta}
        torch.save(deltas, args.out.with_suffix(".delta.pt"))
    else:
        deltas = torch.load(args.edit)
    print(json.dumps(write_adapter(deltas, args.out, args.rank), indent=1))


if __name__ == "__main__":
    main()

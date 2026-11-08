"""LoRA on the int4 QAT checkpoint (compressed-tensors pack-quantized, symmetric, group 32), without
dequantizing the whole model: each Linear keeps its packed int32 weights and bf16 group scales and
rebuilds its bf16 weight on the fly in forward (and again in backward, under gradient checkpointing).
PEFT cannot wrap these modules (they have no .weight). docs/design_changes.md, 2026-11-07.

Saved adapters use PEFT's key names (base_model.model.<module>.lora_A.weight / lora_B.weight) and an
adapter_config.json, so vLLM can load them like a PEFT adapter.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """int32 [out, in/8] -> int8 [out, in] in [-8, 7] (8 nibbles per int32, low bits first, offset 8)."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (nib.reshape(packed.shape[0], -1)[:, :in_features] - 8).to(torch.int8)


class _QMatmul(torch.autograd.Function):
    """x W^T with W rebuilt from int4; backward rebuilds W again instead of keeping it (the base
    weight is frozen, so neither W nor x is saved)."""

    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        return F.linear(x, mod.dequant(x.dtype))

    @staticmethod
    def backward(ctx, gy):
        return gy @ ctx.mod.dequant(gy.dtype), None


class QLinear(nn.Module):
    """y = x W^T (W rebuilt from int4) + (alpha/r) * x A^T B^T. r=0: frozen base only."""

    def __init__(self, base: nn.Module, r: int = 0, alpha: int = 0):
        super().__init__()
        packed = base.weight_packed.data
        scale = base.weight_scale.data
        shape = getattr(base, "weight_shape", None)
        out_f, in_f = (int(shape[0]), int(shape[1])) if shape is not None else (packed.shape[0], packed.shape[1] * 8)
        self.in_features, self.out_features = in_f, out_f
        self.group = in_f // scale.shape[1]
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        b = getattr(base, "bias", None)
        self.register_buffer("bias", None if b is None else b.data, persistent=False)
        self.r = r
        if r:
            self.lora_A = nn.Parameter(torch.empty(r, in_f, dtype=torch.float32, device=packed.device))
            self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.float32, device=packed.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scaling = alpha / r

    def dequant(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """bf16 [out, in], written nibble by nibble (temporaries are 1/8 of the weight, in int32)."""
        o, n = self.out_features, self.in_features
        w = torch.empty(o, self.packed.shape[1] * 8, dtype=self.scale.dtype, device=self.packed.device)
        v = w.view(o, -1, 8)
        for i in range(8):
            v[:, :, i] = ((self.packed >> (4 * i)) & 0xF) - 8
        w = w[:, :n]
        w.view(o, -1, self.group).mul_(self.scale.unsqueeze(-1))
        return w if dtype is None else w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _QMatmul.apply(x, self)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        if self.r:  # bf16 compute, fp32 master weights
            y = y + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        return y


def wrap_model(model: nn.Module, targets: list[str], r: int, alpha: int, scope: str = "language_model") -> list[str]:
    """Replace every packed Linear under `scope` by a QLinear (LoRA on the target projections only)."""
    pat = re.compile(rf"\.({'|'.join(targets)})$")
    names = [n for n, m in model.named_modules() if scope in n and hasattr(m, "weight_packed")]
    lora = []
    for n in names:
        parent_name, child = n.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        use = bool(pat.search(n))
        setattr(parent, child, QLinear(getattr(parent, child), r if use else 0, alpha))
        if use:
            lora.append(n)
    for p in model.parameters():
        p.requires_grad_(False)
    for n, m in model.named_modules():
        if isinstance(m, QLinear) and m.r:
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return lora


def save_adapter(model: nn.Module, out: Path, targets: list[str], r: int, alpha: int, base_name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    sd = {}
    for n, m in model.named_modules():
        if isinstance(m, QLinear) and m.r:
            sd[f"base_model.model.{n}.lora_A.weight"] = m.lora_A.detach().to(torch.bfloat16).contiguous().cpu()
            sd[f"base_model.model.{n}.lora_B.weight"] = m.lora_B.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_name,
        "r": r, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "target_modules": targets, "inference_mode": True,
    }, indent=1))

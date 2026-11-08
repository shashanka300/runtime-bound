"""Track W: read one linear layer of Gemma 4 12B as bf16, dequantised from the QAT int4 checkpoint
exactly as the scorer computes with (int4 symmetric, group 32, pack-quantized). Seconds per matrix,
no model load. Edits are additive deltas against these matrices.

    from trackw.weights import weight
    W = weight(15, "mlp.down_proj")        # [3840, 15360] bf16
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SNAP = next((ROOT / "data/hf/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots").iterdir())
GROUP = 32


@lru_cache(maxsize=1)
def _file():
    return safe_open(str(SNAP / "model.safetensors"), "pt")


def _unpack(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """int32 [rows, cols/8] -> int8 [rows, cols], 4-bit two's-complement offset by 8."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF  # [rows, cols/8, 8]
    return (nib.reshape(packed.shape[0], -1)[:, :cols] - 8).to(torch.int8)


def weight(layer: int, module: str) -> torch.Tensor:
    pre = f"model.language_model.layers.{layer}.{module}."
    f = _file()
    packed = f.get_tensor(pre + "weight_packed")
    scale = f.get_tensor(pre + "weight_scale").float()
    rows, cols = (int(x) for x in f.get_tensor(pre + "weight_shape").tolist())
    q = _unpack(packed, cols).float()
    w = q.view(rows, cols // GROUP, GROUP) * scale.unsqueeze(-1)
    return w.view(rows, cols).to(torch.bfloat16)


if __name__ == "__main__":  # self-check against the transformers-decompressed model
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import acts

    # transformers keeps compressed-tensors linears packed and dequantises inside forward, so
    # compare outputs on random inputs
    # (the weight only materialises inside a full-model forward), so capture each linear's input
    # and output during a short real forward
    model = acts.load("cpu")
    blocks = acts.decoder_layers(model)
    checks = [(0, "mlp.down_proj"), (15, "mlp.gate_proj"), (47, "self_attn.q_proj")]
    seen = {}
    for li, mod in checks:
        blocks[li].get_submodule(mod).register_forward_hook(
            lambda _m, inp, out, key=(li, mod): seen.__setitem__(key, (inp[0], out))
        )
    with torch.no_grad():
        model(input_ids=torch.tensor([[2, 818, 5279, 529, 9405, 563, 236761]]))
    for li, mod in checks:
        x, ref = seen[(li, mod)]
        w = weight(li, mod)
        mine = x.float() @ w.float().T
        rel = float((mine - ref.float()).norm() / ref.float().norm())
        print(
            li,
            mod,
            tuple(w.shape),
            f"relative output diff {rel:.5f}",
            "OK" if rel < 1e-2 else "MISMATCH",
        )

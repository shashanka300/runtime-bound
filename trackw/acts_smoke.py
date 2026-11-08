"""Smoke test for trackw.acts.run hooks (residual + MLP input) on a short prompt, CPU."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts

ids = json.loads(Path(sys.argv[1]).with_suffix(".tokens.jsonl").read_text().splitlines()[0])[
    "prompt_ids"
][:600]
model = acts.load("cpu")
acts.CHUNK = 256
h = acts.run(model, ids, [299, 599], [0, 23, 47])
print("shape", tuple(h.shape))
for li, layer in enumerate([0, 23, 47]):
    r, m = h[1, li, 0], h[1, li, 1]
    print(f"layer {layer}: |resid| {r.norm():.1f} max {r.abs().max():.1f}; |mlp_in| {m.norm():.1f}")

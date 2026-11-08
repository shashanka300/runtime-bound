"""Check trackw.acts.run: chunked prefill (small CHUNK) == one-chunk prefill, on CPU."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts

rec = Path(sys.argv[1]).with_suffix(".tokens.jsonl")
ids = json.loads(rec.read_text().splitlines()[0])["prompt_ids"]
model = acts.load("cpu")
layers = [8, 24, 47]
pos = [len(ids) // 2, len(ids) - 1]
res = {}
for chunk in (1024, 8192):
    acts.CHUNK = chunk
    t0 = time.time()
    res[chunk] = acts.run(model, ids, pos, layers)
    print(f"chunk {chunk}: {time.time() - t0:.0f}s, shape {tuple(res[chunk].shape)}", flush=True)
a, b = res[1024], res[8192]
rel = ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
print(
    f"tokens {len(ids)}; max relative diff {rel:.4f}; min cosine {cos:.5f}: "
    f"{'OK' if cos > 0.999 else 'MISMATCH'}"
)

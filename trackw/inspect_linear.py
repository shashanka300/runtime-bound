"""What does a decompressed compressed-tensors Linear hold in transformers?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acts

model = acts.load("cpu")
m = acts.decoder_layers(model)[15].mlp.down_proj
print(type(m), m)
for n, p in m.named_parameters(recurse=True):
    print("param", n, tuple(p.shape), p.dtype)
for n, b in m.named_buffers(recurse=True):
    print("buffer", n, tuple(b.shape), b.dtype)
print([a for a in dir(m) if "weight" in a or "quant" in a][:30])
print("forward hooks", m._forward_pre_hooks, m._forward_hooks)

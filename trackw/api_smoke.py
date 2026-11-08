"""End-to-end smoke test of the editor API in a throwaway session (data/trackw/selftest)."""

import os
from pathlib import Path
import sys
import time

os.environ["TRACKW_SESSION"] = str(Path(__file__).resolve().parents[1] / "data/trackw/selftest")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from editor_api import acts, budget_left, evaluate, points, save_edit, weight  # noqa: E402

t0 = time.time()
pts = points()
print("dev points", len(pts), "flags example", pts[5]["flags"], f"{time.time() - t0:.0f}s")
ids, layers, h = acts()
print("acts", h.shape, h.dtype, "ids match points:", set(ids) <= {p["i"] for p in pts},
      f"{time.time() - t0:.0f}s")
w = weight(30, "mlp.down_proj")
print("weight", tuple(w.shape), f"{time.time() - t0:.0f}s")
g = torch.Generator().manual_seed(1)
delta = (torch.randn(3840, 1, generator=g) @ torch.randn(1, 15360, generator=g)) * 1e-4
save_edit({"layers.30.mlp.down_proj": delta}, "tiny")
print("budget before", budget_left())
print(evaluate("tiny"))
print("budget after", budget_left(), f"{time.time() - t0:.0f}s")

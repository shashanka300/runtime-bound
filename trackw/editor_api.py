"""Track W editor API: everything the editing agent may use. CPU-only except `evaluate`, which is
the budgeted "forward pass" (50 per session) run on the scorer's engine.

    import sys; sys.path.insert(0, "<repo>/trackw"); from editor_api import *

points()            -> list of decision points (dev split only): tool, args, flags, state, i, task
acts()              -> (point ids, layers, tensor [n_points, 48 layers, 2, 3840] bf16):
                       [:, L, 0] residual stream after layer L, [:, L, 1] MLP input of layer L,
                       at the last prompt token (the moment before the model chooses its action)
weight(L, module)   -> bf16 dequantised matrix, e.g. weight(20, "mlp.down_proj") [3840, 15360]
save_edit(deltas, name) -> writes {"layers.L.<module>": delta [out, in]} and its rank-8 LoRA
evaluate(name, split="dev") -> behaviour on decision points with the edit (costs 1 of 50)
budget_left()       -> remaining evaluations
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eval_points import LEDGER, RECORDS, split_of  # noqa: E402
from lora_export import write_adapter  # noqa: E402
from weights import weight  # noqa: E402,F401

_CUR = ROOT / "data/trackw/CURRENT_SESSION"  # set by the session launcher (env vars do not
# cross the Windows -> wsl boundary)
SESSION = Path(
    __import__("os").environ.get("TRACKW_SESSION")
    or (_CUR.read_text().strip() if _CUR.exists() else ROOT / "data/trackw/session")
)
BUDGET = 50
TARGET = __import__("os").environ.get("TRACKW_TARGET", "reread+dup+regrep")


def points(split: str = "dev") -> list[dict]:
    rows = [json.loads(x) for x in RECORDS.with_suffix(".points.jsonl").read_text().splitlines()]
    return [p for p in rows if split_of(p["task"]) == split]


def acts(split: str = "dev"):
    d = torch.load(RECORDS.with_suffix(".acts.pt"))
    keep = {p["i"] for p in points(split)}
    idx = [n for n, i in enumerate(d["i"]) if i in keep]
    return [d["i"][n] for n in idx], d["layers"], d["h"][idx]


def save_edit(deltas: dict[str, torch.Tensor], name: str, rank: int = 8) -> Path:
    out = SESSION / "edits" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(deltas, out / "deltas.pt")
    report = write_adapter(deltas, out / "lora", rank)
    print(json.dumps(report, indent=1))
    return out / "lora"


def _used() -> int:
    if not LEDGER.exists():
        return 0
    return sum(
        1
        for ln in LEDGER.read_text().splitlines()
        if json.loads(ln).get("adapter", "").startswith(str(SESSION))
    )


def budget_left() -> int:
    return BUDGET - _used()


def evaluate(name: str, split: str = "dev") -> dict:
    if split != "dev":
        raise ValueError("the test split is held out from the editor")
    if budget_left() <= 0:
        raise RuntimeError("evaluation budget exhausted")
    lora = SESSION / "edits" / name / "lora"
    out = subprocess.run(
        [
            sys.executable,
            str(HERE / "eval_points.py"),
            "--target",
            TARGET,
            "--adapter",
            str(lora),
            "--split",
            split,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out[out.index("{") :])

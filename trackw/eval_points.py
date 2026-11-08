"""Track W4: the editor's budgeted "forward pass": re-run recorded decision points on the base
model or on an edit (LoRA hot-loaded into vLLM 0.19.1, the scorer's engine) and score the new
actions with the same rules as the labels (trackw.decision_points.classify).

Needs: vLLM 12B served with --enable-lora --max-lora-rank 8 and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.

    python3 trackw/eval_points.py --target reread --adapter data/trackw/edit_v1 [--split dev]

Points: "target" = recorded actions carrying --target flag; "control" = an equal number of
unflagged points (fixed seed). Splits: tasks hashed into dev (editor may use) / test (held out).
Each run is appended to data/trackw/ledger.jsonl (the editor's budget is 50 runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_points import classify

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data/screen_nov/lb06_12b_requests.jsonl"
LEDGER = ROOT / "data/trackw/ledger.jsonl"
BASE = "gemma-4-12b-it-qat-w4a16-ct"
API = "http://127.0.0.1:8000"


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(raw) if raw.strip().startswith(b"{") else {"text": raw.decode()}


def load_adapter(path: Path) -> str:
    name = "edit_" + hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    models = json.loads(urllib.request.urlopen(API + "/v1/models").read())["data"]
    for m in models:  # max_loras 1: unload whatever is mounted
        if m["id"] != BASE and m.get("parent"):
            _post("/v1/unload_lora_adapter", {"lora_name": m["id"]})
    _post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path.resolve())})
    return name


def split_of(task: str) -> str:
    return "test" if int(hashlib.sha1(task.encode()).hexdigest(), 16) % 3 == 0 else "dev"


def select(points: list[dict], target: str, split: str, n_max: int) -> list[dict]:
    pts = [p for p in points if p["tool"] != "<text>" and split_of(p["task"]) == split]
    tgt = [dict(p, role="target") for p in pts if hits(p["flags"], target)]
    ctl = [p for p in pts if not p["flags"]]
    random.Random(0).shuffle(ctl)
    ctl = [dict(p, role="control") for p in ctl[: len(tgt)]]
    return tgt[:n_max] + ctl[:n_max]


def hits(flags: list[str], target: str) -> bool:
    """target is one flag or a family joined by '+', e.g. 'reread+dup+regrep'."""
    return any(f in flags for f in target.split("+"))


def act(model: str, body: dict, seed: int) -> tuple[str, dict]:
    body = dict(body, model=model, seed=seed)
    resp = _post("/v1/chat/completions", body)
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return "<text>", {}
    f = calls[0]["function"]
    try:
        return f["name"], json.loads(f.get("arguments") or "{}")
    except json.JSONDecodeError:
        return f["name"], {"_raw": f.get("arguments")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="reread+dup+regrep")
    ap.add_argument("--adapter", type=Path, default=None, help="omit for the base model")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-max", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = RECORDS.read_text().splitlines()
    points = [
        json.loads(ln) for ln in RECORDS.with_suffix(".points.jsonl").read_text().splitlines()
    ]
    sel = select(points, args.target, args.split, args.n_max)
    model = load_adapter(args.adapter) if args.adapter else BASE
    t0 = time.time()

    def one(p: dict) -> dict:
        body = json.loads(recs[p["i"]])["request"]
        try:
            name, a = act(model, body, args.seed)
        except Exception as exc:
            return dict(role=p["role"], i=p["i"], error=repr(exc)[:200])
        return dict(
            role=p["role"],
            i=p["i"],
            tool=name,
            args=json.dumps(a)[:160],
            flags=classify(p["state"], name, a),
            recorded_tool=p["tool"],
        )

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, sel))
    tgt = [r for r in rows if r["role"] == "target" and "error" not in r]
    ctl = [r for r in rows if r["role"] == "control" and "error" not in r]
    summary = {
        "t": time.strftime("%FT%TZ", time.gmtime()),
        "adapter": str(args.adapter),
        "target": args.target,
        "split": args.split,
        "seed": args.seed,
        "n_target": len(tgt),
        "n_control": len(ctl),
        "target_rate": round(sum(hits(r["flags"], args.target) for r in tgt) / max(1, len(tgt)), 3),
        "control_flagged": round(sum(bool(r["flags"]) for r in ctl) / max(1, len(ctl)), 3),
        "control_tool_changed": round(
            sum(r["tool"] != r["recorded_tool"] for r in ctl) / max(1, len(ctl)), 3
        ),
        "no_tool_call": round(
            sum(r["tool"] == "<text>" for r in tgt + ctl) / max(1, len(tgt + ctl)), 3
        ),
        "errors": sum("error" in r for r in rows),
        "seconds": round(time.time() - t0),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(dict(summary, rows=rows)) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

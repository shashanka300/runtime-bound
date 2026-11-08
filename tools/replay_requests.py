"""Replay recorded chat-completion requests (tools/record_proxy.py) against whichever vLLM
serves --base, and record per-request completion tokens, reasoning length and latency. The same
recorded conversation prefixes are sent to each engine, so differences in how an engine renders
multi-turn history show up as different outputs for identical inputs.

    uv run python tools/replay_requests.py data/probes/recorded.jsonl v019 [--greedy]
Writes data/probes/replay_<label>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from rbcore.paths import data_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recorded", type=Path)
    ap.add_argument("label")
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--greedy", action="store_true", help="temperature 0 for an engine-only diff")
    args = ap.parse_args()
    out = data_root() / "probes" / f"replay_{args.label}.jsonl"
    rows = [json.loads(line) for line in args.recorded.read_text().splitlines() if line]
    with httpx.Client(timeout=900) as client, out.open("w") as fh:
        for i, rec in enumerate(rows):
            body = dict(rec["request"])
            body.pop("stream", None)
            if args.greedy:
                body["temperature"] = 0.0
            t0 = time.monotonic()
            r = client.post(f"{args.base}/chat/completions", json=body)
            row: dict[str, object] = {
                "i": i,
                "label": args.label,
                "n_messages": len(body.get("messages", [])),
                "status": r.status_code,
                "latency_s": round(time.monotonic() - t0, 3),
            }
            if r.status_code == 200:
                d = r.json()
                msg = d["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                row |= {
                    "prompt_tokens": d["usage"]["prompt_tokens"],
                    "completion_tokens": d["usage"]["completion_tokens"],
                    "reasoning_chars": len(reasoning),
                    "tool_calls": len(msg.get("tool_calls") or []),
                    "finish_reason": d["choices"][0]["finish_reason"],
                }
            else:
                row["error"] = r.text[:300]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    print(out)


if __name__ == "__main__":
    main()

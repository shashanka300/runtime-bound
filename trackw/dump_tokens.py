"""Track W2: store vLLM's own prompt token ids for every recorded request (the scorer renders
through vLLM, whose template output differs from a plain apply_chat_template by a space before
<|tool>). Incremental and idempotent; loops until the record file stops growing for --idle s.

    infra/vllm019/.venv/bin/python trackw/dump_tokens.py data/screen_nov/lb06_12b_requests.jsonl
writes <records>.tokens.jsonl: {"i", "prompt_ids", "usage_prompt_tokens", "ok"}.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def vllm_tokens(body: dict) -> list[int]:
    req = {k: body[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in body}
    req["add_generation_prompt"] = True
    r = urllib.request.Request(
        "http://127.0.0.1:8000/tokenize",
        data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(r, timeout=120).read())["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", type=Path)
    ap.add_argument("--idle", type=int, default=1800)
    args = ap.parse_args()
    out = args.records.with_suffix(".tokens.jsonl")
    done = sum(1 for _ in out.open()) if out.exists() else 0
    last_growth = time.time()
    while True:
        lines = args.records.read_text().splitlines()
        if len(lines) > done:
            last_growth = time.time()
            with out.open("a") as fh:
                for i in range(done, len(lines)):
                    rec = json.loads(lines[i])
                    usage = (rec.get("response") or {}).get("usage") or {}
                    try:
                        ids = vllm_tokens(rec["request"])
                    except Exception as exc:  # server gone: stop, rows so far are kept
                        print(f"stopped at {i}: {exc!r}", flush=True)
                        return
                    fh.write(
                        json.dumps(
                            {
                                "i": i,
                                "prompt_ids": ids,
                                "usage_prompt_tokens": usage.get("prompt_tokens"),
                                "ok": len(ids) == usage.get("prompt_tokens"),
                            }
                        )
                        + "\n"
                    )
                    done = i + 1
            print(f"{done} records tokenised", flush=True)
        if time.time() - last_growth > args.idle:
            print("record file idle; done", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()

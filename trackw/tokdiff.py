"""Diff trackw.replay.render against vLLM's own /tokenize for recorded requests."""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import records, render, tokenizer


def vllm_tokens(body: dict) -> list[int]:
    req = {k: body[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in body}
    req["add_generation_prompt"] = True
    r = urllib.request.Request(
        "http://127.0.0.1:8000/tokenize",
        data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(r, timeout=60).read())["tokens"]


for i, rec in enumerate(records(Path(sys.argv[1]))):
    if i >= int(sys.argv[2] if len(sys.argv) > 2 else 3):
        break
    mine, theirs = render(rec["request"]), vllm_tokens(rec["request"])
    want = rec["response"]["usage"]["prompt_tokens"]
    j = next(
        (k for k, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs))
    )
    tok = tokenizer()
    print(f"record {i}: mine {len(mine)} /tokenize {len(theirs)} usage {want}; first diff at {j}")
    print("   mine  :", repr(tok.decode(mine[max(0, j - 8) : j + 8])))
    print("   theirs:", repr(tok.decode(theirs[max(0, j - 8) : j + 8])))

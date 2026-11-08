"""Track W2: turn recorded agent requests (tools/record_proxy.py JSONL) into token ids exactly as
vLLM 0.19.1 rendered them, so decision points can be re-run on edited weights.

    infra/vllm019/.venv/bin/python trackw/replay.py data/screen_nov/lb06_12b_requests.jsonl
checks rendered length == usage.prompt_tokens for every recorded request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SNAP = next((ROOT / "data/hf/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots").iterdir())
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(SNAP)
    return _TOK


def _normalise(messages: list[dict]) -> list[dict]:
    """vLLM parses tool-call argument strings into dicts before templating."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                args = tc["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        tc["function"]["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def render_text(body: dict) -> str:
    kwargs = dict(body.get("chat_template_kwargs") or {})
    return tokenizer().apply_chat_template(
        _normalise(body["messages"]),
        tools=body.get("tools"),
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )


def render(body: dict, add_special_tokens: bool = False) -> list[int]:
    """vLLM renders the template to text and then encodes it."""
    return tokenizer().encode(render_text(body), add_special_tokens=add_special_tokens)


def records(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    path = Path(sys.argv[1])
    ok = bad = 0
    for i, r in enumerate(records(path)):
        resp = r["response"]
        if not isinstance(resp, dict) or "usage" not in resp:
            continue
        want = resp["usage"]["prompt_tokens"]
        got = len(render(r["request"]))
        if i < 3:
            alt = len(render(r["request"], add_special_tokens=True))
            print(f"record {i}: no-special {got}, with-special {alt}, vLLM {want}")
        if got == want:
            ok += 1
        else:
            bad += 1
            if bad <= 5:
                print(f"record {i}: rendered {got} vs vLLM {want} ({got - want:+d})")
    print(f"exact: {ok}, mismatched: {bad}")


if __name__ == "__main__":
    main()

"""Controlled engine probe: identical first-turn agent requests (starter system prompt + dev40
issue + the 9 tool schemas) sent to whichever vLLM serves :8000, to measure how long the model
thinks per call. Run once per engine and compare (transfer-gap mechanism, 2026-09-27).

    uv run python tools/engine_probe.py <label> [--n 12] [--repeats 2]
Writes data/probes/<label>.jsonl (one row per request).
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from rbcore.paths import competition_dir, data_root
from swebox.tasks import load_tasks, select

TOOLS = [
    ("run_command", {"command": "string"}),
    ("read_file", {"filepath": "string", "start_line": "integer", "end_line": "integer"}),
    ("edit_file", {"filepath": "string", "old_string": "string", "new_string": "string"}),
    ("write_file", {"filepath": "string", "content": "string"}),
    ("get_status", {}),
    ("submit_patch", {}),
    ("get_code_neighbors", {"node": "string", "edge_type": "string", "max_neighbors": "integer"}),
    ("search_similar_code", {"query": "string", "k": "integer"}),
    ("get_code_subgraph", {"nodes": "array"}),
]


def tool_schema(name: str, params: dict[str, str]) -> dict[str, object]:
    props = {
        k: ({"type": "array", "items": {"type": "string"}} if t == "array" else {"type": t})
        for k, t in params.items()
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name.replace("_", " "),
            "parameters": {"type": "object", "properties": props},
        },
    }


# (label, extra request fields): the scorer's official mapping for the starter config, the
# same without reasoning_effort, and greedy decoding for an engine-only comparison.
CONDITIONS = [
    ("official_t0.2", {"temperature": 0.2, "top_p": 0.95, "reasoning_effort": "high"}),
    ("no_effort_t0.2", {"temperature": 0.2, "top_p": 0.95}),
    ("official_greedy", {"temperature": 0.0, "reasoning_effort": "high"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    system = (competition_dir() / "sample_submission" / "prompts" / "system.md").read_text()
    tasks = select(load_tasks(), "dev40")[: args.n]
    tools = [tool_schema(n, p) for n, p in TOOLS]
    out = data_root() / "probes" / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=600) as client, out.open("w") as fh:
        for cond, extra in CONDITIONS:
            for task in tasks:
                for rep in range(args.repeats if extra["temperature"] else 1):
                    body = {
                        "model": "gemma-4-31b-it",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": task.problem_statement},
                        ],
                        "tools": tools,
                        "max_tokens": 8192,
                        "seed": rep,
                        "chat_template_kwargs": {"enable_thinking": True},
                        **extra,
                    }
                    t0 = time.monotonic()
                    r = client.post(f"{args.base}/chat/completions", json=body).json()
                    msg = r["choices"][0]["message"]
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    row = {
                        "label": args.label,
                        "condition": cond,
                        "instance_id": task.instance_id,
                        "rep": rep,
                        "latency_s": round(time.monotonic() - t0, 3),
                        "completion_tokens": r["usage"]["completion_tokens"],
                        "reasoning_chars": len(reasoning),
                        "content_chars": len(msg.get("content") or ""),
                        "tool_calls": len(msg.get("tool_calls") or []),
                        "finish_reason": r["choices"][0]["finish_reason"],
                        "reasoning_head": reasoning[:300],
                        "content_head": (msg.get("content") or "")[:300],
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(cond, task.instance_id, rep, row["completion_tokens"], row["latency_s"])
    print(out)


if __name__ == "__main__":
    main()

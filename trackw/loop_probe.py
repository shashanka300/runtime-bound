"""Track W1b: is the 12B thought-channel loop ("<|channel>thought\\n<channel|>" repeated to the
token limit) caused by the model or by vLLM's thinking_token_budget enforcement?

Replays every recorded request that looped (agent steps and summariser calls) on vLLM 0.19.1 + 12B
under four settings, 3 seeds each, max 4096 new tokens:
  A as recorded (temperature 0.2, thinking on, thinking_token_budget 4096)
  B thinking on, NO thinking_token_budget
  C thinking on, budget 4096, temperature 1.0
  D thinking off (enable_thinking false, no budget)
Reports the loop rate and completion tokens per setting.

    python3 trackw/loop_probe.py data/screen_nov/lb06_12b_requests.jsonl
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOOP = "<|channel>thought\n<channel|><|channel>thought"
SETTINGS = {
    "A_recorded": {},
    "B_no_budget": {"thinking_token_budget": None},
    "C_temp1": {"temperature": 1.0},
    "D_thinking_off": {
        "thinking_token_budget": None,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


def call(body: dict) -> dict:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def looped(resp: dict) -> bool:
    m = resp["choices"][0]["message"]
    return LOOP in ((m.get("content") or "") + (m.get("reasoning") or ""))


def main() -> None:
    path = Path(sys.argv[1])
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    loopers = [
        r["request"]
        for r in recs
        if isinstance(r.get("response"), dict)
        and "choices" in r["response"]
        and looped(r["response"])
    ]
    print(f"{len(loopers)} recorded looping requests", flush=True)
    jobs = []
    for name, over in SETTINGS.items():
        for k, body in enumerate(loopers):
            for seed in range(3):
                b = dict(body, seed=seed, max_completion_tokens=4096)
                b.pop("max_tokens", None)
                for key, val in over.items():
                    if val is None:
                        b.pop(key, None)
                    else:
                        b[key] = val
                jobs.append((name, k, seed, b))

    def run(job):
        name, k, seed, b = job
        try:
            r = call(b)
            return name, looped(r), r["usage"]["completion_tokens"]
        except Exception as exc:  # report, keep going
            return name, None, repr(exc)[:80]

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, jobs))
    for name in SETTINGS:
        rows = [r for r in res if r[0] == name and r[1] is not None]
        toks = sorted(r[2] for r in rows)
        print(
            f"{name:<16} loop {sum(r[1] for r in rows)}/{len(rows)}; completion tokens median "
            f"{toks[len(toks) // 2] if toks else '-'}, max {max(toks) if toks else '-'}"
        )
    out = path.with_suffix(".loop_probe.json")
    out.write_text(json.dumps(res))
    print(out)


if __name__ == "__main__":
    main()

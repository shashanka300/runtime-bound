"""Recording reverse proxy for the OpenAI-compatible vLLM API: forwards every request to the
upstream server unchanged and appends {request, response} to a JSONL file, so a real multi-turn
agent session can be replayed later on another engine (tools/replay_requests.py).

    uv run python tools/record_proxy.py --port 8001 --upstream http://127.0.0.1:8000 \
        --out data/probes/recorded.jsonl
Point the harness at http://127.0.0.1:8001/v1. Single-threaded forwarding per request thread.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

LOCK = threading.Lock()


def make_handler(upstream: str, out: Path) -> type[BaseHTTPRequestHandler]:
    client = httpx.Client(timeout=900)

    class Handler(BaseHTTPRequestHandler):
        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            t0 = time.monotonic()
            r = client.request(
                method,
                upstream + self.path,
                content=body,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            )
            if self.path.endswith("/chat/completions") and body:
                with LOCK, out.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "latency_s": round(time.monotonic() - t0, 3),
                                "request": json.loads(body),
                                "response": r.json() if r.status_code == 200 else r.text,
                                "status": r.status_code,
                            }
                        )
                        + "\n"
                    )
            self.send_response(r.status_code)
            self.send_header("Content-Type", r.headers.get("content-type", "application/json"))
            self.send_header("Content-Length", str(len(r.content)))
            self.end_headers()
            self.wfile.write(r.content)

        def do_POST(self) -> None:
            self._forward("POST")

        def do_GET(self) -> None:
            self._forward("GET")

        def log_message(self, *_: object) -> None:
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.upstream, args.out))
    print(f"proxy :{args.port} -> {args.upstream}, recording to {args.out}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

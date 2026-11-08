"""Switch the long-lived vLLM server between base models (one fits on the GPU at a time).

    ensure_model(ModelSpec(hf_id="google/gemma-4-12B-it", served_name="gemma-4-12b-it"))

If the server already serves ``served_name`` this is a no-op; otherwise the running server
is stopped and ``infra/vllm/serve.sh`` is started detached with the requested model, and
the call blocks until the model is listed by ``/v1/models`` (or startup fails).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil
import structlog

from rbcore.paths import data_root, repo_root

log = structlog.get_logger(__name__)

SERVE_SH = repo_root() / "infra/vllm/serve.sh"


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    served_name: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def served_models(base_url: str) -> set[str]:
    try:
        data = httpx.get(f"{base_url}/v1/models", timeout=5).raise_for_status().json()
    except httpx.HTTPError:
        return set()
    return {m["id"] for m in data["data"]}


def _server_processes() -> list[psutil.Process]:
    """vLLM API server processes started from this repo's serving env."""
    marker = str(repo_root() / "infra/vllm/.venv/bin/vllm")
    out = []
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info.get("cmdline") or []
        if len(cmd) >= 3 and cmd[1] == marker and cmd[2] == "serve":
            out.append(p)
    return out


def stop_server(timeout_s: float = 120) -> None:
    procs = _server_processes()
    family = [c for p in procs for c in p.children(recursive=True)] + procs
    for p in family:
        with contextlib.suppress(psutil.NoSuchProcess):
            p.terminate()
    _, alive = psutil.wait_procs(family, timeout=timeout_s)
    for p in alive:
        p.kill()
    if family:
        log.info("serving.stopped", processes=len(family))


def ensure_model(
    spec: ModelSpec, base_url: str = "http://127.0.0.1:8000", timeout_s: float = 1800
) -> None:
    if spec.served_name in served_models(base_url):
        return
    stop_server()
    env = os.environ | {"MODEL": spec.hf_id}
    args = ["--served-model-name", spec.served_name, *spec.extra_args]
    log_dir = data_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("vllm_*.log"))
    subprocess.Popen(
        ["setsid", "-f", "bash", str(SERVE_SH), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=repo_root(),
    )
    log.info("serving.starting", model=spec.hf_id, served=spec.served_name)
    t0 = time.monotonic()
    new_log: Path | None = None
    while time.monotonic() - t0 < timeout_s:
        time.sleep(5)
        if spec.served_name in served_models(base_url):
            log.info("serving.ready", served=spec.served_name, seconds=round(time.monotonic() - t0))
            return
        new_log = new_log or next(iter(set(log_dir.glob("vllm_*.log")) - before), None)
        if new_log and "Engine core initialization failed" in new_log.read_text(errors="replace"):
            raise RuntimeError(f"vLLM failed to start {spec.hf_id}; see {new_log}")
    raise TimeoutError(f"vLLM did not serve {spec.served_name} within {timeout_s}s")

"""Run manifest: everything needed to say exactly what produced a number.

Collected once at run start and written to ``runs/<run_id>/manifest.json``. Every
probe is best-effort: a missing tool is recorded as ``None``, never an exception.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from rbcore.config import ExperimentConfig
from rbcore.paths import data_root, repo_root

KEY_PACKAGES = (
    "rbcore",
    "numpy",
    "scipy",
    "polars",
    "pyarrow",
    "duckdb",
    "matplotlib",
    "pydantic",
    "httpx",
    "nvidia-ml-py",
    "mlflow-skinny",
    "opentelemetry-sdk",
    "codecarbon",
    "google-adk",
    "papermill",
)


def collect(cfg: ExperimentConfig[Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": cfg.experiment,
        "config_hash": cfg.hash(),
        "seed": cfg.seed,
        "sandbox": {**cfg.sandbox.model_dump(), "matches_eval": cfg.sandbox.matches_eval},
        "git": _safe(_git),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {p: _version(p) for p in KEY_PACKAGES},
        "uv_lock_sha256": _safe(lambda: _sha256(repo_root() / "uv.lock")),
        "serving_lock_sha256": _safe(lambda: _sha256(repo_root() / "infra/vllm/uv.lock")),
        "host": _host(),
        "gpu": _safe(_gpus),
        "docker": _safe(_docker),
        "vllm": _safe(lambda: _vllm(cfg.serving.base_url)),
        "vllm_server_env": _safe(_vllm_server_env),
        "data_manifest": _safe(lambda: _sha256(data_root() / "DATA_MANIFEST.json")),
    }


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        return {"error": repr(exc)}


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _host() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "wsl": "microsoft" in platform.release().lower(),
        "cpu_model": cpu_model,
        "logical_cpus": psutil.cpu_count(logical=True),
        "mem_total_gib": round(psutil.virtual_memory().total / 2**30, 1),
    }


def _gpus() -> list[dict[str, Any]]:
    from rbcore.hwmon.nvml import gpu_inventory

    return gpu_inventory()


def _docker() -> dict[str, Any]:
    import docker

    client = docker.from_env()
    try:
        v = client.version()
        info = client.info()
        return {
            "version": v.get("Version"),
            "cgroup_version": info.get("CgroupVersion"),
            "cgroup_driver": info.get("CgroupDriver"),
            "runtimes": sorted(info.get("Runtimes", {})),
        }
    finally:
        client.close()


def _vllm_server_env() -> dict[str, str] | None:
    """Model, CLI overrides and VLLM_* env of the running server (written by serve.sh)."""
    path = data_root() / "logs" / "vllm_latest.env"
    if not path.exists():
        return None
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _vllm(base_url: str) -> dict[str, Any]:
    with httpx.Client(base_url=base_url, timeout=2.0) as c:
        version = c.get("/version").raise_for_status().json().get("version")
        models = c.get("/v1/models").raise_for_status().json()["data"]
    return {
        "version": version,
        "models": [
            {"id": m["id"], "root": m.get("root"), "max_model_len": m.get("max_model_len")}
            for m in models
        ],
    }

"""Hardware and serving telemetry. See :mod:`rbcore.hwmon.base` for the sampling model."""

from __future__ import annotations

import importlib

import structlog

from rbcore.config import MonitorConfig, ServingConfig
from rbcore.hwmon.base import Monitor, Sampler

log = structlog.get_logger(__name__)

__all__ = ["Monitor", "Sampler", "build_samplers"]


def build_samplers(cfg: MonitorConfig, serving: ServingConfig) -> list[Sampler]:
    """Instantiate every enabled sampler; unavailable sources are logged and skipped."""
    samplers: list[Sampler] = []
    wanted = {
        "nvml": ("rbcore.hwmon.nvml", "NvmlSampler", {}),
        "host": ("rbcore.hwmon.host", "HostSampler", {}),
        "docker": ("rbcore.hwmon.docker_stats", "DockerSampler", {}),
        "vllm": ("rbcore.hwmon.vllm_metrics", "VllmMetricsSampler", {"base_url": serving.base_url}),
    }
    for key, (module, cls, kwargs) in wanted.items():
        if not getattr(cfg, key):
            continue
        try:
            samplers.append(getattr(importlib.import_module(module), cls)(**kwargs))
        except Exception as exc:
            log.warning("hwmon.sampler_unavailable", sampler=key, error=repr(exc))
    return samplers

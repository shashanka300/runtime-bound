"""GPU telemetry via NVML (nvidia-ml-py). Works under WSL2; DCGM does not.

Energy comes from the driver's cumulative counter (mJ since driver load), which is
exact, unlike integrating sampled power. Per-task energy = counter delta over the
task's [start, end] window. Fields the driver does not support on this platform are
detected once at startup and skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pynvml as nv

from rbcore.hwmon.base import Row, Sampler

_FIELDS: dict[str, Callable[[Any], Any]] = {
    "util_gpu_pct": lambda h: nv.nvmlDeviceGetUtilizationRates(h).gpu,
    "util_mem_pct": lambda h: nv.nvmlDeviceGetUtilizationRates(h).memory,
    "mem_used_mib": lambda h: nv.nvmlDeviceGetMemoryInfo(h).used / 2**20,
    "power_w": lambda h: nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
    "energy_mj": nv.nvmlDeviceGetTotalEnergyConsumption,
    "temp_c": lambda h: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
    "sm_clock_mhz": lambda h: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
    "mem_clock_mhz": lambda h: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
    "pstate": nv.nvmlDeviceGetPerformanceState,
    "throttle_reasons": nv.nvmlDeviceGetCurrentClocksEventReasons,
    "fan_pct": nv.nvmlDeviceGetFanSpeed,
}


class NvmlSampler(Sampler):
    name = "gpu"

    def __init__(self) -> None:
        nv.nvmlInit()
        self._handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        self._fields = {
            k: f for k, f in _FIELDS.items() if all(_supported(f, h) for h in self._handles)
        }

    @property
    def unsupported(self) -> list[str]:
        return sorted(set(_FIELDS) - set(self._fields))

    def sample(self) -> list[Row]:
        rows = []
        for i, h in enumerate(self._handles):
            row: Row = {"gpu": i}
            for k, f in self._fields.items():
                row[k] = f(h)
            rows.append(row)
        return rows

    def close(self) -> None:
        nv.nvmlShutdown()


def _supported(f: Callable[[Any], Any], handle: Any) -> bool:
    try:
        f(handle)
    except nv.NVMLError:
        return False
    return True


def gpu_inventory() -> list[dict[str, Any]]:
    """Static GPU facts for the run manifest."""
    nv.nvmlInit()
    try:
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            info: dict[str, Any] = {
                "index": i,
                "name": nv.nvmlDeviceGetName(h),
                "uuid": nv.nvmlDeviceGetUUID(h),
                "mem_total_mib": nv.nvmlDeviceGetMemoryInfo(h).total / 2**20,
            }
            for key, fn in {
                "power_limit_w": lambda h: nv.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                "compute_capability": lambda h: ".".join(
                    map(str, nv.nvmlDeviceGetCudaComputeCapability(h))
                ),
                "max_sm_clock_mhz": lambda h: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
                "vbios": nv.nvmlDeviceGetVbiosVersion,
            }.items():
                try:
                    info[key] = fn(h)
                except nv.NVMLError:
                    info[key] = None
            out.append(info)
        return [
            {
                "driver": nv.nvmlSystemGetDriverVersion(),
                "cuda_driver": nv.nvmlSystemGetCudaDriverVersion(),
            },
            *out,
        ]
    finally:
        nv.nvmlShutdown()

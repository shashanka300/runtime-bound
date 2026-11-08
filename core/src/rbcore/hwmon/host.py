"""Host CPU / memory / disk telemetry (psutil). Rates are deltas between samples."""

from __future__ import annotations

import time

import psutil

from rbcore.hwmon.base import Row, Sampler


class HostSampler(Sampler):
    name = "host"

    def __init__(self) -> None:
        psutil.cpu_percent(None)  # prime: first call returns 0.0
        self._disk = psutil.disk_io_counters()
        self._t = time.monotonic()

    def sample(self) -> list[Row]:
        now = time.monotonic()
        dt = max(now - self._t, 1e-6)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        disk = psutil.disk_io_counters()
        load1, _, _ = psutil.getloadavg()
        row: Row = {
            "cpu_pct": psutil.cpu_percent(None),
            "load1": load1,
            "mem_used_gib": (vm.total - vm.available) / 2**30,
            "mem_avail_gib": vm.available / 2**30,
            "swap_used_gib": sw.used / 2**30,
        }
        if disk and self._disk:
            row["disk_read_mbps"] = (disk.read_bytes - self._disk.read_bytes) / dt / 2**20
            row["disk_write_mbps"] = (disk.write_bytes - self._disk.write_bytes) / dt / 2**20
        self._disk, self._t = disk, now
        return [row]

"""Per-container sandbox telemetry read straight from cgroup files.

``docker stats`` costs ~1-2 s per container per call, far too slow for 1 Hz sampling
of several sandboxes. cgroup files are the same source docker reads, at microsecond
cost. The container list is refreshed through the Docker API every few seconds.

Both cgroup v2 (unified) and v1 (the WSL2 5.15 kernel default, cgroupfs driver) are
supported. Rows: container name, memory (current / peak), CPU% since last sample
(100 = one core) and cumulative ``oom_kill`` count (exit 137 in the harness).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rbcore.hwmon.base import Row, Sampler

CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class CgroupStats:
    cpu_usec: int
    mem_bytes: int
    peak_bytes: int | None
    oom_kills: int


def _read_kv(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition(" ")
        if v.strip().isdigit():
            out[k] = int(v)
    return out


def _read_int(path: Path) -> int | None:
    return int(path.read_text()) if path.exists() else None


class CgroupV2:
    """Unified hierarchy: one directory per container."""

    def __init__(self, root: Path = CGROUP_ROOT) -> None:
        self.root = root

    def locate(self, cid: str) -> Path | None:
        for d in (self.root / "system.slice" / f"docker-{cid}.scope", self.root / "docker" / cid):
            if (d / "cpu.stat").exists():
                return d
        return None

    def read(self, d: Path) -> CgroupStats:
        return CgroupStats(
            cpu_usec=_read_kv(d / "cpu.stat")["usage_usec"],
            mem_bytes=int((d / "memory.current").read_text()),
            peak_bytes=_read_int(d / "memory.peak"),
            oom_kills=_read_kv(d / "memory.events").get("oom_kill", 0),
        )


class CgroupV1:
    """Legacy per-controller hierarchies; ``d`` is the container's memory cgroup dir."""

    def __init__(self, root: Path = CGROUP_ROOT) -> None:
        self.root = root
        cpu = next((root / n for n in ("cpuacct", "cpu,cpuacct") if (root / n).is_dir()), None)
        self.cpu_root = cpu or root / "cpuacct"

    def locate(self, cid: str) -> Path | None:
        mem = self.root / "memory"
        for d in (mem / "docker" / cid, mem / "system.slice" / f"docker-{cid}.scope"):
            if (d / "memory.usage_in_bytes").exists():
                return d
        return None

    def read(self, d: Path) -> CgroupStats:
        rel = d.relative_to(self.root / "memory")
        oom_file = d / "memory.oom_control"
        return CgroupStats(
            cpu_usec=int((self.cpu_root / rel / "cpuacct.usage").read_text()) // 1000,
            mem_bytes=int((d / "memory.usage_in_bytes").read_text()),
            peak_bytes=_read_int(d / "memory.max_usage_in_bytes"),
            oom_kills=_read_kv(oom_file).get("oom_kill", 0) if oom_file.exists() else 0,
        )


def detect(root: Path = CGROUP_ROOT) -> CgroupV1 | CgroupV2:
    return CgroupV2(root) if (root / "cgroup.controllers").exists() else CgroupV1(root)


class DockerSampler(Sampler):
    name = "containers"

    def __init__(self, label_filter: str | None = None, refresh_s: float = 5.0) -> None:
        import docker

        self._client = docker.from_env()
        self._client.ping()
        self._cg = detect()
        self._label = label_filter
        self._refresh_s = refresh_s
        self._last_refresh = -float("inf")
        self._containers: dict[str, tuple[str, Path]] = {}  # id -> (name, cgroup dir)
        self._cpu_prev: dict[str, tuple[int, float]] = {}

    def _refresh(self) -> None:
        filters = {"label": self._label} if self._label else {}
        current = {}
        for c in self._client.containers.list(filters=filters):
            if (d := self._cg.locate(c.id)) is not None:
                current[c.id] = (c.name, d)
        self._containers = current
        self._cpu_prev = {k: v for k, v in self._cpu_prev.items() if k in current}
        self._last_refresh = time.monotonic()

    def sample(self) -> list[Row]:
        if time.monotonic() - self._last_refresh >= self._refresh_s:
            self._refresh()
        now = time.monotonic()
        return [
            row
            for cid, (name, d) in list(self._containers.items())
            if (row := self._row(cid, name, d, now)) is not None
        ]

    def _row(self, cid: str, name: str, d: Path, now: float) -> Row | None:
        try:
            s = self._cg.read(d)
        except (FileNotFoundError, ProcessLookupError, KeyError):
            self._containers.pop(cid, None)  # container exited between refreshes
            return None
        prev = self._cpu_prev.get(cid)
        self._cpu_prev[cid] = (s.cpu_usec, now)
        cpu_pct = None
        if prev is not None:
            cpu_pct = (s.cpu_usec - prev[0]) / 1e6 / max(now - prev[1], 1e-6) * 100
        return {
            "container": name,
            "cpu_pct": cpu_pct,
            "mem_mib": s.mem_bytes / 2**20,
            "mem_peak_mib": None if s.peak_bytes is None else s.peak_bytes / 2**20,
            "oom_kills": s.oom_kills,
        }

    def close(self) -> None:
        self._client.close()

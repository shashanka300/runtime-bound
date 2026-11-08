"""Low-overhead background telemetry.

One daemon thread polls every registered :class:`Sampler` at a fixed interval and
buffers rows in memory; buffers are flushed as numbered Parquet part files
(``hw/<sampler>/part-00000.parquet``) so a crash loses at most one flush window and
nothing is ever rewritten. Read back with ``polars.scan_parquet(run_dir/"hw/<name>/*.parquet")``.

Every row carries ``t_ns`` (wall clock, UTC epoch ns) so samples join against run
events and OTel spans, which use the same clock.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
import structlog

log = structlog.get_logger(__name__)

Row = dict[str, Any]


class Sampler(ABC):
    """A source of telemetry rows. ``sample`` must be fast (well under the interval)."""

    name: str

    @abstractmethod
    def sample(self) -> list[Row]: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class Monitor:
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(
        self,
        out_dir: Path,
        samplers: list[Sampler],
        interval_s: float = 1.0,
        flush_every_s: float = 30.0,
    ) -> None:
        self.out_dir = out_dir
        self.samplers = samplers
        self.interval_s = interval_s
        self.flush_every_s = flush_every_s
        self._buffers: dict[str, list[Row]] = {s.name: [] for s in samplers}
        self._parts: dict[str, int] = dict.fromkeys(self._buffers, 0)
        self._errors: dict[str, int] = dict.fromkeys(self._buffers, 0)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.overhead_by: dict[str, float] = dict.fromkeys(self._buffers, 0.0)
        """Seconds spent inside each sampler's sample(); reported so overhead is auditable."""

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> Monitor:
        if self._thread is not None:
            raise RuntimeError("monitor already started")
        # Synchronous baseline sample: guarantees a t0 row (energy counters are deltas).
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, name="rbcore-hwmon", daemon=True)
        self._thread.start()
        log.info(
            "hwmon.start", samplers=[s.name for s in self.samplers], interval_s=self.interval_s
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 5)
        self.flush()
        for s in self.samplers:
            try:
                s.close()
            except Exception:
                log.exception("hwmon.close_failed", sampler=s.name)
        log.info("hwmon.stop", overhead_s={k: round(v, 3) for k, v in self.overhead_by.items()})

    @property
    def overhead_s(self) -> float:
        return sum(self.overhead_by.values())

    # -- sampling --------------------------------------------------------------
    def _loop(self) -> None:
        last_flush = time.monotonic()
        next_tick = last_flush + self.interval_s
        while not self._stop.wait(max(0.0, next_tick - time.monotonic())):
            self.sample_once()
            now = time.monotonic()
            if now - last_flush >= self.flush_every_s:
                self.flush()
                last_flush = now
            next_tick += self.interval_s
            # Drift-free schedule; if we fell behind, skip ticks rather than burst.
            if next_tick < now:
                next_tick = now + self.interval_s

    def sample_once(self) -> None:
        for sampler in self.samplers:
            if self._errors[sampler.name] >= self.MAX_CONSECUTIVE_ERRORS:
                continue
            t0 = time.perf_counter()
            try:
                rows = sampler.sample()
            except Exception as exc:
                self._errors[sampler.name] += 1
                level = (
                    "error"
                    if self._errors[sampler.name] >= self.MAX_CONSECUTIVE_ERRORS
                    else "warning"
                )
                getattr(log, level)(
                    "hwmon.sample_failed",
                    sampler=sampler.name,
                    error=repr(exc),
                    disabled=level == "error",
                )
                continue
            finally:
                self.overhead_by[sampler.name] += time.perf_counter() - t0
            self._errors[sampler.name] = 0
            if rows:
                t_ns = time.time_ns()
                for r in rows:
                    r.setdefault("t_ns", t_ns)
                with self._lock:
                    self._buffers[sampler.name].extend(rows)

    def flush(self) -> None:
        with self._lock:
            pending = {k: v for k, v in self._buffers.items() if v}
            for k in pending:
                self._buffers[k] = []
        for name, rows in pending.items():
            d = self.out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"part-{self._parts[name]:05d}.parquet"
            self._parts[name] += 1
            pl.from_dicts(rows, infer_schema_length=None).write_parquet(path)

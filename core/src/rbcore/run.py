"""RunContext: the one object every experiment notebook opens.

    with RunContext(cfg, exp_dir=EXP_DIR) as run:
        with run.phase("sweep", concurrency=4):
            ...
        run.log_metrics({"tok_s": 812.0})

On enter it creates ``runs/<run_id>/``, writes the resolved config and manifest,
starts JSON logging, tracing, hardware monitors and CodeCarbon, and opens an MLflow
run. On exit (even on error) it flushes everything and writes ``summary.json``.

Run directory layout::

    config.yaml  manifest.json  summary.json
    logs.jsonl   events.jsonl   metrics.jsonl   spans.jsonl
    hw/{gpu,host,containers,vllm}/part-*.parquet
    emissions.csv   artifacts/
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
import polars as pl
import structlog
import yaml

from rbcore import logging as rblog
from rbcore import manifest, tracing
from rbcore.config import ExperimentConfig
from rbcore.hwmon import Monitor, build_samplers

log = structlog.get_logger(__name__)


class RunContext:
    def __init__(
        self,
        cfg: ExperimentConfig[Any],
        exp_dir: str | Path,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.exp_dir = Path(exp_dir).resolve()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{cfg.hash()[:8]}-{secrets.token_hex(2)}"
        self.run_name = run_name or self.run_id
        self.tags = tags or {}
        self.dir = self.exp_dir / "runs" / self.run_id
        self.artifacts = self.dir / "artifacts"
        self.monitor: Monitor | None = None
        self._mlflow = False
        self._carbon: Any = None
        self._log_handler: Any = None
        self._t0 = 0.0

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> Self:
        rblog.configure()
        self.artifacts.mkdir(parents=True, exist_ok=False)
        self._t0 = time.monotonic()
        self._log_handler = rblog.add_run_file(self.dir / "logs.jsonl")
        structlog.contextvars.bind_contextvars(run_id=self.run_id, experiment=self.cfg.experiment)

        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(self.cfg.model_dump(mode="json"), sort_keys=False)
        )
        self.manifest = manifest.collect(self.cfg, self.run_id)
        _write_json(self.dir / "manifest.json", self.manifest)
        if not self.cfg.sandbox.matches_eval:
            log.warning(
                "run.sandbox_not_eval_faithful",
                profile=self.cfg.sandbox.profile,
                note="fine for development; headline numbers need profile=eval",
            )

        tracing.start_run_tracing(self.dir / "spans.jsonl", self.cfg.tracking.phoenix_endpoint)
        self._start_mlflow()
        self._start_monitors()
        self.event("run_start")
        log.info("run.start", dir=str(self.dir))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        status = "FINISHED" if exc is None else "FAILED"
        if exc is not None:
            log.error("run.failed", error=repr(exc))
        self.event("run_end", status=status)
        overhead: dict[str, float] = {}
        if self.monitor is not None:
            self.monitor.stop()
            overhead = {k: round(v, 3) for k, v in self.monitor.overhead_by.items()}
        emissions = self._stop_carbon()
        tracing.stop_run_tracing()

        summary = {
            "run_id": self.run_id,
            "status": status,
            "wall_s": round(time.monotonic() - self._t0, 3),
            "monitor_overhead_s": overhead,
            "gpu_energy_kwh": self._gpu_energy_kwh(),
            "codecarbon": emissions,
        }
        _write_json(self.dir / "summary.json", summary)
        self._end_mlflow(status, summary)
        log.info("run.end", **{k: v for k, v in summary.items() if k != "codecarbon"})
        structlog.contextvars.unbind_contextvars("run_id", "experiment")
        rblog.remove_handler(self._log_handler)

    # Notebook-friendly aliases: a run that spans several cells is opened with start()
    # and closed in the teardown cell with finish(); scripts should use ``with``.
    def start(self) -> Self:
        return self.__enter__()

    def finish(self, error: BaseException | None = None) -> None:
        self.__exit__(type(error) if error else None, error, None)

    # -- events, phases, metrics ------------------------------------------------
    def event(self, name: str, **attrs: Any) -> None:
        """Append a timestamped event (wall clock ns, same clock as hw samples and spans)."""
        rec = {"t_ns": time.time_ns(), "event": name, **attrs}
        with (self.dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    @contextmanager
    def phase(self, name: str, **attrs: Any) -> Iterator[Any]:
        """Bracket a unit of work: start/end events + an OTel span with the same attributes.

        Hardware samples inside [start, end] are attributed to it by ``window()``.
        """
        pid = secrets.token_hex(6)  # pairs start/end even when phases overlap (concurrency)
        self.event("phase_start", phase=name, phase_id=pid, **attrs)
        with tracing.tracer().start_as_current_span(name, attributes=_otel_attrs(attrs)) as span:
            try:
                yield span
            except BaseException as exc:
                self.event(
                    "phase_end", phase=name, phase_id=pid, status="error", error=repr(exc), **attrs
                )
                raise
        self.event("phase_end", phase=name, phase_id=pid, status="ok", **attrs)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        rec = {"t_ns": time.time_ns(), "step": step, **metrics}
        with (self.dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self._mlflow:
            import mlflow

            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow:
            import mlflow

            mlflow.log_artifact(str(path))

    # -- reading back live data (for inline notebook plots) ----------------------
    def hw(self, source: str) -> pl.DataFrame:
        """Samples collected so far for ``source`` in {gpu, host, containers, vllm}."""
        if self.monitor is not None:
            self.monitor.flush()
        return read_hw(self.dir, source)

    def events(self) -> pl.DataFrame:
        return read_events(self.dir)

    # -- internals ---------------------------------------------------------------
    def _start_monitors(self) -> None:
        m = self.cfg.monitor
        samplers = build_samplers(m, self.cfg.serving)
        if samplers:
            self.monitor = Monitor(self.dir / "hw", samplers, m.interval_s, m.flush_every_s).start()
        if m.codecarbon:
            try:
                from codecarbon import EmissionsTracker, OfflineEmissionsTracker

                common: dict[str, Any] = {
                    "output_dir": str(self.dir),
                    "log_level": "error",
                    "save_to_file": True,
                    "project_name": self.run_id,
                    "tracking_mode": "machine",
                }
                self._carbon = (
                    OfflineEmissionsTracker(country_iso_code=m.codecarbon_country, **common)
                    if m.codecarbon_country
                    else EmissionsTracker(**common)
                )
                self._carbon.start()
            except Exception as exc:
                log.warning("run.codecarbon_unavailable", error=repr(exc))
                self._carbon = None

    def _stop_carbon(self) -> dict[str, Any] | None:
        if self._carbon is None:
            return None
        try:
            self._carbon.stop()
            d = self._carbon.final_emissions_data
            return {
                "energy_kwh": d.energy_consumed,
                "emissions_kg": d.emissions,
                "cpu_kwh": d.cpu_energy,
                "gpu_kwh": d.gpu_energy,
                "ram_kwh": d.ram_energy,
                "region": d.country_iso_code,
            }
        except Exception as exc:
            log.warning("run.codecarbon_stop_failed", error=repr(exc))
            return None

    def _gpu_energy_kwh(self) -> float | None:
        try:
            gpu = read_hw(self.dir, "gpu")
        except FileNotFoundError:
            return None
        if "energy_mj" not in gpu.columns or gpu.is_empty():
            return None
        per_gpu = gpu.group_by("gpu").agg(pl.col("energy_mj").max() - pl.col("energy_mj").min())
        return float(per_gpu["energy_mj"].sum()) / 3.6e9

    def _start_mlflow(self) -> None:
        uri = self.cfg.tracking.mlflow_uri
        if not uri:
            return
        try:
            httpx.get(f"{uri}/health", timeout=0.5).raise_for_status()
        except httpx.HTTPError:
            log.warning("run.mlflow_unreachable", uri=uri, hint="bash infra/services.sh start")
            return
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.cfg.experiment)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params({k: str(v)[:6000] for k, v in self.cfg.flat().items()})
        git = self.manifest.get("git") or {}
        mlflow.set_tags(
            {
                "run_id": self.run_id,
                "config_hash": self.cfg.hash(),
                "git_sha": str(git.get("sha")),
                "git_dirty": str(git.get("dirty")),
                "sandbox_profile": self.cfg.sandbox.profile,
                "run_dir": str(self.dir),
                **self.tags,
            }
        )
        self._mlflow = True

    def _end_mlflow(self, status: str, summary: dict[str, Any]) -> None:
        if not self._mlflow:
            return
        import mlflow

        for name in ("config.yaml", "manifest.json", "summary.json", "events.jsonl"):
            if (self.dir / name).exists():
                mlflow.log_artifact(str(self.dir / name))
        numeric = {k: v for k, v in summary.items() if isinstance(v, int | float)}
        numeric |= {f"monitor_overhead_s.{k}": v for k, v in summary["monitor_overhead_s"].items()}
        mlflow.log_metrics(numeric)
        mlflow.end_run(status=status)


# -- reading finished runs (used by replay experiments such as E2/E6) -----------------
def read_hw(run_dir: str | Path, source: str) -> pl.DataFrame:
    parts = sorted((Path(run_dir) / "hw" / source).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no {source} samples under {run_dir}")
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed").with_columns(
        pl.from_epoch("t_ns", time_unit="ns").alias("time")
    )


def read_events(run_dir: str | Path) -> pl.DataFrame:
    path = Path(run_dir) / "events.jsonl"
    return pl.read_ndjson(path, infer_schema_length=None).with_columns(
        pl.from_epoch("t_ns", time_unit="ns").alias("time")
    )


def phase_windows(events: pl.DataFrame) -> pl.DataFrame:
    """One row per phase occurrence: its start attributes plus [t_start_ns, t_end_ns, status].

    Start and end are paired by ``phase_id``, so overlapping phases (concurrent tasks) are
    attributed correctly. A phase that is still running has a null ``t_end_ns``.
    """
    starts = (
        events.filter(pl.col("event") == "phase_start")
        .drop("event", "time", strict=False)
        .rename({"t_ns": "t_start_ns"})
    )
    starts = starts.drop([c for c in ("status", "error") if c in starts.columns])
    ends = events.filter(pl.col("event") == "phase_end").select(
        "phase_id", pl.col("t_ns").alias("t_end_ns"), "status"
    )
    return starts.join(ends, on="phase_id", how="left").drop("phase_id")


def _otel_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    ok = (str, bool, int, float)
    return {k: v if isinstance(v, ok) else json.dumps(v, default=str) for k, v in attrs.items()}


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))

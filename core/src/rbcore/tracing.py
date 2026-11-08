"""OpenTelemetry tracing: every span goes to ``runs/<run_id>/spans.jsonl`` and, if the
local Phoenix collector is reachable, to Phoenix for browsing.

The OTel global TracerProvider can be set only once per process, while notebooks start
many runs in one kernel. So the provider is created once and its exporters point at a
*switchable* target that each run swaps in and out.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import httpx
import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

log = structlog.get_logger(__name__)


class _SwitchableExporter(SpanExporter):
    """Forwards to whatever exporter the current run installed (or drops spans)."""

    def __init__(self) -> None:
        self.target: SpanExporter | None = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        target = self.target
        return target.export(spans) if target else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class JsonlSpanExporter(SpanExporter):
    """One compact OTel-JSON span per line; the source of truth for offline analysis."""

    def __init__(self, path: Path) -> None:
        self._fh: IO[str] = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for s in spans:
                self._fh.write(json.dumps(json.loads(s.to_json()), separators=(",", ":")) + "\n")
            self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._fh.close()


class _Fanout(SpanExporter):
    def __init__(self, exporters: list[SpanExporter]) -> None:
        self.exporters = exporters

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        results = [e.export(spans) for e in self.exporters]
        ok = all(r == SpanExportResult.SUCCESS for r in results)
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        for e in self.exporters:
            e.shutdown()


_provider: TracerProvider | None = None
_switch = _SwitchableExporter()
_processor: BatchSpanProcessor | None = None


def _ensure_provider() -> TracerProvider:
    global _provider, _processor
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": "runtime-bound"}))
        _processor = BatchSpanProcessor(_switch, schedule_delay_millis=1000)
        _provider.add_span_processor(_processor)
        trace.set_tracer_provider(_provider)
        _instrument_adk(_provider)
    return _provider


def _instrument_adk(provider: TracerProvider) -> None:
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    except ImportError:
        return
    GoogleADKInstrumentor().instrument(tracer_provider=provider)
    log.info("tracing.adk_instrumented")


def start_run_tracing(spans_path: Path, phoenix_endpoint: str | None) -> None:
    _ensure_provider()
    exporters: list[SpanExporter] = [JsonlSpanExporter(spans_path)]
    if phoenix_endpoint and _reachable(phoenix_endpoint):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporters.append(OTLPSpanExporter(endpoint=phoenix_endpoint))
    elif phoenix_endpoint:
        log.warning(
            "tracing.phoenix_unreachable",
            endpoint=phoenix_endpoint,
            hint="bash infra/services.sh start",
        )
    _switch.target = _Fanout(exporters)


def stop_run_tracing() -> None:
    if _processor is not None:
        _processor.force_flush()
    target, _switch.target = _switch.target, None
    if target is not None:
        target.shutdown()


def tracer(name: str = "rbcore") -> trace.Tracer:
    return _ensure_provider().get_tracer(name)


def _reachable(endpoint: str) -> bool:
    try:
        httpx.get(endpoint.rsplit("/v1/", 1)[0], timeout=0.5)
    except httpx.HTTPError:
        return False
    return True

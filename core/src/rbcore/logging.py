"""Structured logging: human-readable in the notebook, JSON lines on disk.

Configured once per process (idempotent, safe to re-run notebook cells). A run adds a
file handler for ``runs/<run_id>/logs.jsonl`` and binds ``run_id`` via contextvars so
every line, from any module, is attributable to the run.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

_PRE_CHAIN: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]
_configured = False


def configure(level: int = logging.INFO) -> None:
    """Root logger at ``level`` (what run files receive). The console shows WARNING+ unless
    ``RB_CONSOLE_LEVEL`` says otherwise: long notebook runs must not stream every INFO line
    into cell output (it bloats the saved notebook until autosave stalls the run)."""
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[*_PRE_CHAIN, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(os.environ.get("RB_CONSOLE_LEVEL", "WARNING").upper())
    # Plain tracebacks: the default rich formatter prints every frame's locals (whole LLM
    # sessions here), which once grew a notebook's output past 50 GB of RAM (E1, 2026-09-25).
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(
                colors=False, exception_formatter=structlog.dev.plain_traceback
            ),
            foreign_pre_chain=_PRE_CHAIN,
        )
    )
    root = logging.getLogger()
    root.handlers = [console]
    root.setLevel(level)
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "docker",
        "matplotlib",
        "PIL",
        "fontTools",
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Benign under asyncio: an OTel context token detached from another task after a
    # generator is closed. Logged at ERROR with a traceback on every early-closed stream.
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
    _quiet_litellm()
    _configured = True


def _quiet_litellm() -> None:
    """LiteLLM prints one INFO line per call to stdout via its own handler."""
    try:
        import litellm
    except ImportError:
        return
    litellm.suppress_debug_info = True


def add_run_file(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=_PRE_CHAIN,
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()

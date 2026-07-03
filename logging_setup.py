"""Structured logging for the Nornikel Knowledge Graph.

Two output formats:
    * ``text`` (default) — human-friendly colored output for terminals
    * ``json`` — single-line JSON record per log entry, for log
      aggregators (Loki, ELK, etc.)

The logger respects the ``LOG_LEVEL`` env var and a per-module
level override (``LOG_LEVEL_<MODULE>=DEBUG``).

A ``request_id`` is attached to log records via contextvar so
that every log line in a request handler can be correlated
without the caller having to thread it explicitly.
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _REQUEST_ID.set(request_id)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


# ---------------------------------------------------------------- formatters


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extras = getattr(record, "extra_fields", None)
        if isinstance(extras, dict):
            for k, v in extras.items():
                if k not in payload:
                    payload[k] = v
        return _json.dumps(payload, ensure_ascii=False, default=str)


# ------------------------------------------------------------------- factory


_configured = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_dir: str | None = None,
) -> None:
    """Configure the root logger for the application.

    Idempotent — safe to call multiple times. Tests that need to
    reconfigure should call ``shutdown_logging()`` first.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        # Just adjust level/format and return.
        root.setLevel(level.upper())
        for handler in root.handlers:
            handler.setFormatter(_make_formatter(fmt))
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_make_formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level.upper())

    if log_dir:
        from pathlib import Path
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
        file_handler.setFormatter(_make_formatter("json"))
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers by default.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    # Per-module overrides: ``LOG_LEVEL_<MODULE>=DEBUG`` bumps the level.
    for key, value in os.environ.items():
        if key.startswith("LOG_LEVEL_") and key != "LOG_LEVEL":
            module = key[len("LOG_LEVEL_"):].lower().replace("__", ".")
            logging.getLogger(module).setLevel(value.upper())

    _configured = True


def _make_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        return JsonFormatter()
    return _TextFormatter()


def shutdown_logging() -> None:
    global _configured
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover
            pass
    _configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root logger if not done yet."""
    if not _configured:
        configure_logging(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            fmt=os.environ.get("LOG_FORMAT", "text"),
            log_dir=os.environ.get("LOG_DIR"),
        )
    return logging.getLogger(name)


# ------------------------------------------------------------------ helpers


def log_with(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a message with structured ``fields`` attached.

    Usage::

        log_with(log, logging.INFO, "graph_lookup", seeds=3, max_hop=4)
    """
    logger.log(level, message, extra={"extra_fields": fields})


@dataclass
class Timer:
    """Simple context manager that logs elapsed time on exit."""

    logger: logging.Logger
    operation: str
    level: int = logging.INFO
    extra: dict[str, Any] | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.perf_counter() - self._start
        fields = {"operation": self.operation, "elapsed_sec": round(elapsed, 4)}
        if self.extra:
            fields.update(self.extra)
        if exc_type is not None:
            fields["error"] = f"{exc_type.__name__}: {exc_val}"
        log_with(self.logger, self.level, f"{self.operation} done", **fields)


__all__ = [
    "JsonFormatter",
    "Timer",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "log_with",
    "set_request_id",
    "shutdown_logging",
]

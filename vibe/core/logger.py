from __future__ import annotations

from datetime import UTC, datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import re

from vibe.core.paths import LOG_DIR, LOG_FILE

logger = logging.getLogger("vibe")


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        ppid = os.getppid()
        pid = os.getpid()
        level = record.levelname
        message = encode_log_message(record.getMessage())

        line = f"{timestamp} {ppid} {pid} {level} {message}"

        if record.exc_info:
            exc_text = encode_log_message(self.formatException(record.exc_info))
            line = f"{line} {exc_text}"

        return line


def encode_log_message(message: str) -> str:
    return message.replace("\\", "\\\\").replace("\n", "\\n")


def decode_log_message(encoded: str) -> str:
    return re.sub(
        r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), encoded
    )


def apply_logging_config(target_logger: logging.Logger) -> None:
    max_bytes = int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024))

    # DEBUG_MODE is the debugpy switch (see vibe/acp/entrypoint.py);
    # it also forces DEBUG-level logging here.
    if os.environ.get("DEBUG_MODE") == "true":
        log_level_str = "DEBUG"
    else:
        log_level_str = os.environ.get("LOG_LEVEL", "WARNING").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level_str not in valid_levels:
            log_level_str = "WARNING"

    # backupCount must be > 0: with 0, RotatingFileHandler never truncates on
    # rollover (rename-to-same-path is a no-op), so the file grows unbounded.
    try:
        LOG_DIR.path.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_FILE.path, maxBytes=max_bytes, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredLogFormatter())
        handler.setLevel(logging.WARNING)
        target_logger.setLevel(logging.DEBUG)
        target_logger.addHandler(handler)
        target_logger.warning(
            "File logging is unavailable at %s; using stderr: %s", LOG_FILE.path, exc
        )
        return
    handler.setFormatter(StructuredLogFormatter())
    log_level = getattr(logging, log_level_str, logging.WARNING)
    handler.setLevel(log_level)

    # Make sure the logger is not gating logs
    target_logger.setLevel(logging.DEBUG)

    target_logger.addHandler(handler)


apply_logging_config(logger)

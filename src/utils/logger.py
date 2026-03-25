"""Centralised logging setup for the NSE Analytics platform."""

import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _utf8_stream(stream) -> io.TextIOWrapper:
    """
    Return a UTF-8 text stream wrapping the given stream's binary buffer.
    Falls back to the original stream if reconfiguration is not possible.
    Unknown characters are replaced with '?' instead of raising an exception.
    """
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
            return stream
        if hasattr(stream, "buffer"):
            return io.TextIOWrapper(
                stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass
    return stream


# Reconfigure stdout/stderr once at import time so that any print() calls
# in downloader/transformer modules also work on Windows cp1252 terminals.
sys.stdout = _utf8_stream(sys.stdout)
sys.stderr = _utf8_stream(sys.stderr)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with both console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers if called twice
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler - writes to the already-reconfigured UTF-8 stdout
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(ch)

    # Rotating file handler (10 MB x 5 backups) - explicit UTF-8 encoding
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "nse_analytics.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(fh)

    return logger
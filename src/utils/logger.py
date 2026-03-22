"""Centralised logging setup for the NSE Analytics platform."""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with both console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers if called twice
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(ch)

    # Rotating file handler (10 MB × 5 backups)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "nse_analytics.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    fh.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(fh)

    return logger
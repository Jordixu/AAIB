"""Centralised logging setup for all scripts."""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    """Create a logger that writes DEBUG to file and INFO to console.

    Log file: <log_dir>/<name>_YYYYMMDD_HHMMSS.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log: %s", log_path)
    return logger

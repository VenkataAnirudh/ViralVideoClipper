"""
Pipeline Logger
================
Creates per-job loggers that write to both console AND a debug.log file
inside the job's output directory. Every pipeline module uses this.

Usage:
    from pipeline.logger import create_job_logger
    logger = create_job_logger(job_id, job_dir)
    logger.info("Starting download...")
    logger.debug("Detailed info here")
    logger.error("Something failed", exc_info=True)  # includes traceback
"""

import logging
import os
import sys
import config


class ImmediateFileHandler(logging.FileHandler):
    """File handler that flushes each record all the way to disk."""

    def emit(self, record):
        super().emit(record)
        if getattr(config, "LOG_FSYNC_EACH_RECORD", True):
            try:
                self.flush()
                os.fsync(self.stream.fileno())
            except OSError:
                pass


def create_debug_file_handler(log_file: str, formatter: logging.Formatter) -> logging.Handler:
    """Create the strict debug.log handler used by all pipeline entry points."""
    file_handler = ImmediateFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    return file_handler


def create_job_logger(job_id: str, job_dir: str) -> logging.Logger:
    """
    Create a logger for a specific job that writes to:
      1. Console (stdout) — so you see it in the terminal
      2. File (outputs/<job_id>/debug.log) — for post-mortem debugging

    Args:
        job_id: Unique job identifier
        job_dir: Path to the job's output directory

    Returns:
        logging.Logger configured for this job
    """
    # Create logger with unique name per job
    logger = logging.getLogger(f"viralclipper.{job_id}")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.DEBUG))

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(module)-15s │ %(message)s",
        datefmt="%H:%M:%S"
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console shows INFO+
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler ---
    os.makedirs(job_dir, exist_ok=True)
    log_file = os.path.join(job_dir, "debug.log")
    file_handler = create_debug_file_handler(log_file, formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized for job {job_id}")
    logger.debug(f"Log file: {log_file}")

    return logger


def get_log_contents(job_dir: str, last_n_lines: int = 50) -> str:
    """Read the last N lines of a job's debug log."""
    log_file = os.path.join(job_dir, "debug.log")
    if not os.path.exists(log_file):
        return ""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-last_n_lines:])
    except Exception:
        return ""

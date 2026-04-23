"""
Structured logging module with console + rotating file output.

Why RotatingFileHandler?
    Prevents the log file growing forever on a long-running server.
    When the file hits max_bytes it is renamed app.log.1, app.log.2 etc.
    backup_count controls how many rotated files are kept before deletion.
"""
import logging
import logging.handlers   # ← MUST be here — used in __init__ below
import sys
from pathlib import Path
from typing import Optional


class StructuredLogger:
    """Structured logger with console and file outputs."""
    
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    
    def __init__(
        self,
        name: str = "rag_system",
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 5_242_880,   # 5 MB
        backup_count: int = 3,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers so re-initialisation does not duplicate output
        self.logger.handlers = []

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler — always present
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, level.upper()))
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # Rotating file handler — only when log_file is provided
        if log_file:
            self.LOG_DIR.mkdir(exist_ok=True)
            log_path = self.LOG_DIR / log_file
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    # ── Logging methods ───────────────────────────────────────────
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, **kwargs)

    def event(self, event_name: str, **details):
        """
        Log a structured event with key=value pairs.

        Usage:
            logger.event("tool_selected", intent="quiz", tool="generate_quiz")

        Produces:
            2024-01-01 12:00:00 | INFO     | rag_system | EVENT: tool_selected | intent=quiz | tool=generate_quiz
        """
        details_str = " | ".join(f"{k}={v}" for k, v in details.items())
        self.logger.info(f"EVENT: {event_name} | {details_str}")


# ── Global singleton ──────────────────────────────────────────────────────────
_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "rag_system",
    level: str = "INFO",
    log_file: Optional[str] = "app.log",
) -> StructuredLogger:
    """
    Return the global logger, creating it on first call.

    All modules call get_logger() with no arguments.
    main.py calls it first with the level from general_settings,
    so every subsequent call gets the already-configured instance.
    """
    global _logger
    if _logger is None:
        _logger = StructuredLogger(name=name, level=level, log_file=log_file)
    return _logger

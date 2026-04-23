"""
Structured logging module with console + file outputs.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class StructuredLogger:
    """Structured logger with console and file outputs."""
    
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    
    def __init__(
        self,
        name: str = "rag_system",
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 5242880,  # 5MB
        backup_count: int = 3,
    ):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : str
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file : str, optional
            Log file path relative to logs/ directory
        max_bytes : int
            Max size of log file before rotation
        backup_count : int
            Number of backup logs to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            self.LOG_DIR.mkdir(exist_ok=True)
            log_path = self.LOG_DIR / log_file
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, **kwargs)
    
    def event(self, event_name: str, **details):
        """Log a structured event."""
        details_str = " | ".join(f"{k}={v}" for k, v in details.items())
        self.logger.info(f"EVENT: {event_name} | {details_str}")


# Global logger instance
_logger = None


def get_logger(
    name: str = "rag_system",
    level: str = "INFO",
    log_file: Optional[str] = "app.log",
) -> StructuredLogger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(name=name, level=level, log_file=log_file)
    return _logger


# Import logging.handlers so the RotatingFileHandler can be used
import logging.handlers

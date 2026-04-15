"""
utils/logger.py
---------------
Logging setup for the entire project.
Call setup_logger() once at startup; then use get_logger() anywhere.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config


def setup_logger() -> None:
    """
    Configure the root logger.

    Creates the log directory if it does not exist,
    attaches a rotating file handler and a stream handler.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Rotating file handler (max 5 MB, keep 3 backups)
    if config.LOG_FILE:
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

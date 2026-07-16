"""Logging configuration for the Causal Churn Intervention System.

Call ``setup_logging()`` once at application entry-point (``retrain.py``,
``dashboard/app.py``) to activate structured console logging.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler.

    Safe to call multiple times — subsequent calls are no-ops.

    Args:
        level: Logging level (default ``logging.INFO``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Always prefer this over ``logging.getLogger``
    so that module names are consistent across the project.

    Args:
        name: Logger name — typically ``__name__`` from the calling module.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)

"""Shared utility helpers for the Causal Churn Intervention System."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from src.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator that logs the wall-clock duration of a function call.

    Example::

        @timed
        def train_model(X, y):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2f s", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]

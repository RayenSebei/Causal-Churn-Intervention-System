"""Training entry points for the churn model.

This module provides a stable public API over the legacy ``src.model``
implementation so that future refactors can move training responsibilities
without breaking callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.model import (
    BaselineMetrics,
    build_classifier,
    build_preprocessor,
    load_featured_frame,
    make_pipeline,
    split_features_targets,
    train_baseline_model,
)

__all__ = [
    "BaselineMetrics",
    "build_classifier",
    "build_preprocessor",
    "load_featured_frame",
    "make_pipeline",
    "split_features_targets",
    "train_baseline_model",
]


def train(
    csv_path: str | Path,
    *,
    model_output_path: str | Path,
    calibration_output_path: str | Path,
    random_state: int = 42,
) -> tuple[BaselineMetrics, dict[str, Any]]:
    """Train the baseline churn model.

    This is a thin wrapper over :func:`src.model.train_baseline_model` that
    gives the project a clean `training.py` entry point without changing the
    runtime behavior.
    """

    return train_baseline_model(
        csv_path,
        model_output_path=model_output_path,
        calibration_output_path=calibration_output_path,
        random_state=random_state,
    )

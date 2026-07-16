"""Backward-compatibility shim — training logic now lives in ``training.py``.

Existing imports such as ``from src.model import train_baseline_model`` continue
to work. New code should import from ``src.training`` directly.
"""

from src.training import (  # noqa: F401
    BaselineMetrics,
    build_classifier,
    build_preprocessor,
    load_featured_frame,
    make_pipeline,
    run_cli,
    split_features_targets,
    train,
    train_baseline_model,
)

__all__ = [
    "BaselineMetrics",
    "build_classifier",
    "build_preprocessor",
    "load_featured_frame",
    "make_pipeline",
    "run_cli",
    "split_features_targets",
    "train",
    "train_baseline_model",
]

if __name__ == "__main__":
    run_cli()

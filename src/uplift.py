"""Backward-compatibility shim — uplift logic now lives in ``src.causal``.

Existing imports such as ``from src.uplift import run_uplift_pipeline`` continue
to work. New code should import from ``src.causal`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.causal import (
    assign_synthetic_treatment,
    estimate_cate_from_baseline,
    fit_t_learner_cate,
    inject_treatment_effect,
    predict_t_learner_cate,
    run_uplift_pipeline,
    segment_customers_by_cate,
)
from src.config import paths
from src.constants import TARGET_COLUMN
from src.training import load_featured_frame

__all__ = [
    "assign_synthetic_treatment",
    "inject_treatment_effect",
    "estimate_cate_from_baseline",
    "fit_t_learner_cate",
    "predict_t_learner_cate",
    "segment_customers_by_cate",
    "run_uplift_pipeline",
    "run_cli",
]


def run_cli() -> None:
    """Convenience entry point for uplift modeling."""

    csv_path = paths.raw_csv
    model_path = paths.baseline_model

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=[TARGET_COLUMN])
    y = featured_df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = joblib.load(model_path)

    results = run_uplift_pipeline(X_test, y_test, model)

    print("\n=== Segment Counts ===")
    print(results["segment_counts"])

    print("\n=== CATE Recovery (vs synthetic ground truth) ===")
    print(f"Correlation: {results['cate_recovery_corr']:.3f}")
    print(f"MAE: {results['cate_recovery_mae']:.4f}")

    print("\n=== Segment Examples (Highest Uplift per Segment) ===")
    for seg, ex in results["segment_examples"].items():
        print(
            f"\n{seg}:"
            f"\n  Baseline Churn: {ex['baseline_churn']:.2%}"
            f"\n  Uplift from Treatment: {ex['uplift']:.2%}"
            f"\n  Expected Churn if Treated: {ex['expected_churn_if_treated']:.2%}"
        )


if __name__ == "__main__":
    run_cli()

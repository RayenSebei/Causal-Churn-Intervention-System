"""Data-quality validation for dashboard and causal pipelines."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    ALL_SEGMENTS,
    CUSTOMER_ID_COLUMN,
    FALLBACK_SEGMENT,
    LEGACY_SEGMENT_ALIASES,
    TARGET_SEGMENTS,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

CRITICAL_COLUMNS = [
    "churn_probability",
    "uplift",
    "expected_churn_if_treated",
    "segment",
]


def clip_probability(values: np.ndarray | pd.Series | float) -> np.ndarray:
    """Force values into the valid probability range [0, 1]."""

    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0.0, 1.0)


def treated_churn_probability(
    baseline: np.ndarray | pd.Series | float,
    cate: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Compute expected churn if treated.

    Assumption
    ----------
    ``cate`` is the treatment effect on churn probability
    (positive = churn reduced, negative = backfire / churn increased).

    ``P(churn | treat) = clip(P(churn | control) - cate, 0, 1)``

    This keeps treated probabilities inside a valid probability simplex even
    when CATE estimates are noisy or negative.
    """

    baseline_arr = clip_probability(baseline)
    cate_arr = np.asarray(cate, dtype=float)
    result = clip_probability(baseline_arr - cate_arr)
    return np.atleast_1d(result)


def normalize_segment_labels(segments: pd.Series) -> pd.Series:
    """Map legacy aliases and fill invalid labels with the fallback segment."""

    normalized = segments.astype(object).copy()
    normalized = normalized.replace(LEGACY_SEGMENT_ALIASES)
    invalid = ~normalized.isin(ALL_SEGMENTS) | normalized.isna()
    if invalid.any():
        n_bad = int(invalid.sum())
        logger.warning(
            "Replacing %s invalid/missing segment label(s) with fallback '%s'",
            n_bad,
            FALLBACK_SEGMENT,
        )
        warnings.warn(
            f"{n_bad} invalid segment labels replaced with '{FALLBACK_SEGMENT}'",
            RuntimeWarning,
            stacklevel=2,
        )
        normalized.loc[invalid] = FALLBACK_SEGMENT
    return normalized


def summarize_segment_distribution(segments: pd.Series) -> dict[str, int]:
    """Return counts for every canonical segment (missing keys → 0)."""

    counts = segments.value_counts(dropna=False)
    summary = {seg: int(counts.get(seg, 0)) for seg in ALL_SEGMENTS}
    summary["total"] = int(len(segments))
    return summary


def validate_probabilities(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Return human-readable issues for out-of-range probability columns."""

    columns = columns or ["churn_probability", "expected_churn_if_treated"]
    issues: list[str] = []
    for col in columns:
        if col not in df.columns:
            issues.append(f"Missing probability column: {col}")
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            issues.append(f"{col} contains NaN values")
        if (values < 0).any() or (values > 1).any():
            issues.append(f"{col} has values outside [0, 1]")
    return issues


def validate_segments(df: pd.DataFrame) -> list[str]:
    """Return issues for missing / unknown segment labels."""

    issues: list[str] = []
    if "segment" not in df.columns:
        return ["Missing segment column"]
    if df["segment"].isna().any():
        issues.append("segment contains NaN values")
    unknown = ~df["segment"].isin(ALL_SEGMENTS)
    if unknown.any():
        issues.append(f"segment has {int(unknown.sum())} unknown labels")
    return issues


def validate_uplift_bounds(
    uplift: np.ndarray | pd.Series,
    *,
    min_cate: float,
    max_cate: float,
) -> list[str]:
    """Return issues when uplift leaves the configured CATE bounds."""

    values = pd.to_numeric(pd.Series(uplift), errors="coerce")
    issues: list[str] = []
    if values.isna().any():
        issues.append("uplift contains NaN values")
    if (values < min_cate).any() or (values > max_cate).any():
        issues.append(f"uplift has values outside [{min_cate}, {max_cate}]")
    return issues


def validate_roi_metrics(metrics: dict[str, Any]) -> list[str]:
    """Sanity-check ROI / campaign metric dictionary."""

    issues: list[str] = []
    for key in ("cost_targeted", "cost_blanket", "campaign_cost_targeted", "campaign_cost_blanket"):
        if key in metrics and metrics[key] < 0:
            issues.append(f"{key} is negative")
    for key in ("customers_targeted", "customers_blanket", "expected_customers_retained_targeted"):
        if key in metrics and metrics[key] < 0:
            issues.append(f"{key} is negative")
    return issues


def validate_segment_count_consistency(df: pd.DataFrame) -> list[str]:
    """Ensure segment value counts sum to the dataframe length."""

    issues: list[str] = []
    if "segment" not in df.columns:
        return ["Missing segment column"]
    counted = int(df["segment"].value_counts(dropna=False).sum())
    if counted != len(df):
        msg = f"Segment count mismatch: value_counts sum={counted}, rows={len(df)}"
        issues.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
    return issues


def validate_shap_alignment(df: pd.DataFrame) -> list[str]:
    """Ensure every row has a non-empty SHAP explanation string."""

    issues: list[str] = []
    if "shap_explanation" not in df.columns:
        return ["Missing shap_explanation column"]
    missing = df["shap_explanation"].isna() | df["shap_explanation"].astype(str).str.strip().eq("")
    missing |= df["shap_explanation"].astype(str).eq("No explanation available")
    if missing.any():
        issues.append(f"shap_explanation missing for {int(missing.sum())} rows")
    return issues


def validate_dashboard_frame(
    df: pd.DataFrame,
    *,
    min_cate: float,
    max_cate: float,
    roi_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full dashboard validation suite.

    Returns a report dict with ``ok`` flag, ``issues`` list, and segment summary.
    """

    issues: list[str] = []
    for col in CRITICAL_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing critical column: {col}")

    if not issues:
        issues.extend(validate_probabilities(df))
        issues.extend(validate_segments(df))
        issues.extend(validate_uplift_bounds(df["uplift"], min_cate=min_cate, max_cate=max_cate))
        issues.extend(validate_segment_count_consistency(df))
        issues.extend(validate_shap_alignment(df))

    if roi_metrics is not None:
        issues.extend(validate_roi_metrics(roi_metrics))

    if CUSTOMER_ID_COLUMN in df.columns and df[CUSTOMER_ID_COLUMN].isna().any():
        issues.append(f"{CUSTOMER_ID_COLUMN} contains NaN values")

    segment_summary = (
        summarize_segment_distribution(df["segment"]) if "segment" in df.columns else {}
    )

    report = {
        "ok": len(issues) == 0,
        "issues": issues,
        "segment_summary": segment_summary,
        "target_segments": list(TARGET_SEGMENTS),
        "n_rows": len(df),
    }
    if issues:
        for issue in issues:
            logger.warning("Dashboard validation issue: %s", issue)
            warnings.warn(issue, RuntimeWarning, stacklevel=2)
    else:
        logger.info("Dashboard validation passed for %s rows", len(df))
    return report

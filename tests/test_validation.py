"""Tests for validation utilities and probability/segment guards."""

import numpy as np
import pandas as pd
import pytest

from src.config import causal as causal_config
from src.constants import (
    ALL_SEGMENTS,
    SEGMENT_LOW_RISK_UPSIDE,
    SEGMENT_PERSUADABLES,
    SEGMENT_SLEEPING_DOGS,
)
from src.validation import (
    clip_probability,
    normalize_segment_labels,
    summarize_segment_distribution,
    treated_churn_probability,
    validate_dashboard_frame,
    validate_probabilities,
    validate_segments,
    validate_uplift_bounds,
)


def test_clip_probability_bounds():
    values = clip_probability(np.array([-0.2, 0.5, 1.4]))
    assert values.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_treated_churn_probability_with_backfire():
    treated = treated_churn_probability([0.884], [-0.314])
    assert treated[0] <= 1.0
    assert treated[0] == pytest.approx(1.0)


def test_normalize_legacy_low_risk_upside():
    segments = pd.Series([SEGMENT_LOW_RISK_UPSIDE, SEGMENT_SLEEPING_DOGS, None])
    normalized = normalize_segment_labels(segments)
    assert normalized.iloc[0] == SEGMENT_PERSUADABLES
    assert normalized.isna().sum() == 0
    assert set(normalized).issubset(set(ALL_SEGMENTS))


def test_validate_dashboard_frame_catches_bad_probs():
    df = pd.DataFrame(
        {
            "churn_probability": [0.5, 1.2],
            "uplift": [0.1, 0.1],
            "expected_churn_if_treated": [0.4, 0.5],
            "segment": [SEGMENT_PERSUADABLES, SEGMENT_PERSUADABLES],
            "shap_explanation": ["a", "b"],
        }
    )
    issues = validate_probabilities(df)
    assert any("outside [0, 1]" in issue for issue in issues)


def test_validate_uplift_bounds():
    issues = validate_uplift_bounds(
        np.array([0.1, 0.9]),
        min_cate=causal_config.cate_clip_min,
        max_cate=causal_config.cate_clip_max,
    )
    assert issues


def test_validate_segments_rejects_nan():
    df = pd.DataFrame({"segment": [SEGMENT_PERSUADABLES, np.nan]})
    issues = validate_segments(df)
    assert issues


def test_summarize_segment_distribution_includes_all_keys():
    segments = pd.Series([SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS])
    summary = summarize_segment_distribution(segments)
    assert summary["total"] == 2
    for seg in ALL_SEGMENTS:
        assert seg in summary


def test_validate_dashboard_frame_ok_path():
    df = pd.DataFrame(
        {
            "churn_probability": [0.5, 0.2],
            "uplift": [0.1, -0.05],
            "expected_churn_if_treated": [0.4, 0.25],
            "segment": [SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS],
            "shap_explanation": ["feat A", "feat B"],
            "customerID": ["1", "2"],
        }
    )
    report = validate_dashboard_frame(
        df,
        min_cate=causal_config.cate_clip_min,
        max_cate=causal_config.cate_clip_max,
    )
    assert report["ok"] is True
    assert report["n_rows"] == 2

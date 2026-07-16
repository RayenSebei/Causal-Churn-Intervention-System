"""Unit tests for causal simulation, segmentation, and ROI helpers."""

import numpy as np
import pandas as pd
import pytest

from src.causal import (
    SimulationConfig,
    TreatmentPolicy,
    assign_synthetic_treatment,
    estimate_cate_from_baseline,
    inject_treatment_effect,
    segment_customers_by_cate,
)
from src.causal.roi import expected_campaign_cost, expected_profit, expected_retained_revenue
from src.config import causal as causal_config
from src.constants import ALL_SEGMENTS, SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS
from src.validation import treated_churn_probability


def test_assign_synthetic_treatment_is_binary_and_seeded():
    df = pd.DataFrame(
        {
            "MonthlyCharges": [20.0, 90.0, 40.0, 100.0],
            "Contract": ["One year", "Month-to-month", "Two year", "Month-to-month"],
        }
    )
    a = assign_synthetic_treatment(df, policy=TreatmentPolicy(random_state=0))
    b = assign_synthetic_treatment(df, policy=TreatmentPolicy(random_state=0))
    assert a.tolist() == b.tolist()
    assert set(a.unique()).issubset({0, 1})


def test_inject_treatment_effect_respects_negative_cate():
    baseline = np.array([0.10, 0.40])
    treatment = np.array([1, 1])
    cate = np.array([-0.03, 0.20])
    outcomes, true_cate = inject_treatment_effect(
        baseline, treatment, cate, config=SimulationConfig(random_state=1)
    )
    assert true_cate.tolist() == pytest.approx(cate.tolist())
    assert outcomes.shape == (2,)
    assert set(np.unique(outcomes)).issubset({0, 1})


def test_estimate_cate_from_baseline_can_be_negative_for_long_tenure():
    X = pd.DataFrame({"tenure": [60, 3]})
    baseline = np.array([0.05, 0.80])
    cate = estimate_cate_from_baseline(model=None, X_features=X, baseline_probs=baseline)
    assert cate[0] < 0
    assert cate[1] > 0
    assert cate.min() >= causal_config.cate_clip_min
    assert cate.max() <= causal_config.cate_clip_max


def test_segment_customers_by_cate_is_exhaustive_four_way():
    X = pd.DataFrame({"x": np.arange(8)})
    baseline = np.array([0.1, 0.1, 0.8, 0.8, 0.2, 0.7, 0.9, 0.05])
    cate = np.array([0.01, 0.40, 0.02, 0.50, -0.05, 0.10, 0.60, 0.55])
    segments = segment_customers_by_cate(X, baseline, cate)
    assert segments.isna().sum() == 0
    assert set(segments.unique()).issubset(set(ALL_SEGMENTS))
    assert (segments == SEGMENT_SLEEPING_DOGS).sum() >= 1
    assert "distribution" in segments.attrs
    assert segments.attrs["distribution"]["total"] == 8


def test_segment_handles_nan_cate_without_nan_labels():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    baseline = np.array([0.2, 0.8, 0.3, 0.7])
    cate = np.array([0.1, np.nan, -0.05, 0.2])
    segments = segment_customers_by_cate(X, baseline, cate)
    assert segments.isna().sum() == 0
    assert set(segments.unique()).issubset(set(ALL_SEGMENTS))


def test_treated_probability_never_exceeds_one_with_negative_cate():
    baseline = np.array([0.884, 0.50])
    cate = np.array([-0.40, 0.90])  # will be clipped by caller in pipeline; raw here
    treated = treated_churn_probability(baseline, cate)
    assert treated.min() >= 0.0
    assert treated.max() <= 1.0
    assert treated[0] == pytest.approx(1.0)


def test_uplift_bounds_config():
    assert causal_config.cate_clip_min == pytest.approx(-0.20)
    assert causal_config.cate_clip_max == pytest.approx(0.30)


def test_roi_helpers_basic_math():
    revenue = expected_retained_revenue(
        monthly_charges=np.array([50.0, 80.0]),
        baseline_churn=np.array([0.4, 0.5]),
        treatment_effect=np.array([0.1, 0.2]),
        acceptance_probability=1.0,
        annual_revenue_multiplier=12,
    )
    cost = expected_campaign_cost(10.0, customers_targeted=2, acceptance_probability=1.0)
    profit = expected_profit(revenue, cost)
    assert revenue > 0
    assert cost == 20.0
    assert profit == revenue - cost


def test_persuadables_include_former_low_risk_upside_quadrant():
    X = pd.DataFrame({"x": [0, 1]})
    baseline = np.array([0.1, 0.9])  # low, high
    cate = np.array([0.25, 0.25])  # both high cate
    segments = segment_customers_by_cate(X, baseline, cate, baseline_percentile=50, cate_percentile=50)
    assert (segments == SEGMENT_PERSUADABLES).all()

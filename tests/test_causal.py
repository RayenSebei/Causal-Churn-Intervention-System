"""Unit tests for causal simulation, segmentation, and ROI helpers."""

import numpy as np
import pandas as pd

from src.causal import (
    SimulationConfig,
    TreatmentPolicy,
    assign_synthetic_treatment,
    estimate_cate_from_baseline,
    inject_treatment_effect,
    segment_customers_by_cate,
)
from src.causal.roi import expected_campaign_cost, expected_profit, expected_retained_revenue
from src.constants import (
    ALL_SEGMENTS,
    SEGMENT_SLEEPING_DOGS,
)


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
    assert true_cate.tolist() == cate.tolist()
    assert outcomes.shape == (2,)
    assert set(np.unique(outcomes)).issubset({0, 1})


def test_estimate_cate_from_baseline_can_be_negative_for_long_tenure():
    X = pd.DataFrame({"tenure": [60, 3]})
    baseline = np.array([0.05, 0.80])
    cate = estimate_cate_from_baseline(model=None, X_features=X, baseline_probs=baseline)
    assert cate[0] < 0
    assert cate[1] > 0


def test_segment_customers_by_cate_is_exhaustive():
    X = pd.DataFrame({"x": np.arange(8)})
    baseline = np.array([0.1, 0.1, 0.8, 0.8, 0.2, 0.7, 0.9, 0.05])
    cate = np.array([0.01, 0.40, 0.02, 0.50, -0.05, 0.10, 0.60, 0.55])
    segments = segment_customers_by_cate(X, baseline, cate)
    assert segments.isna().sum() == 0
    assert set(segments.unique()).issubset(set(ALL_SEGMENTS))
    assert (segments == SEGMENT_SLEEPING_DOGS).sum() >= 1


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

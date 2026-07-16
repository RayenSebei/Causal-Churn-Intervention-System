"""Unit tests for dashboard helpers."""

import pandas as pd

from dashboard.filters import customer_selector_options, filter_dashboard_frame
from dashboard.metrics import compute_roi_metrics
from src.constants import SEGMENT_LOW_RISK_UPSIDE, SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": [
                SEGMENT_PERSUADABLES,
                SEGMENT_LOW_RISK_UPSIDE,
                SEGMENT_SLEEPING_DOGS,
                SEGMENT_PERSUADABLES,
            ],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "churn_probability": [0.6, 0.2, 0.1, 0.7],
            "uplift": [0.2, 0.15, -0.03, 0.25],
            "expected_churn_if_treated": [0.4, 0.05, 0.13, 0.45],
            "MonthlyCharges": [70.0, 50.0, 40.0, 90.0],
        }
    )


def test_filter_dashboard_frame_by_segment_and_contract():
    df = _sample_frame()
    filtered = filter_dashboard_frame(
        df, segment_filter=SEGMENT_PERSUADABLES, contract_filter="Month-to-month"
    )
    assert len(filtered) == 2
    assert set(filtered["segment"]) == {SEGMENT_PERSUADABLES}


def test_customer_selector_options_use_index_values():
    df = _sample_frame()
    options = customer_selector_options(df.iloc[[2]])
    assert options[0]["value"] == 2


def test_compute_roi_metrics_excludes_sleeping_dogs():
    df = _sample_frame()
    metrics = compute_roi_metrics(df, discount_cost_per_customer=10.0)
    assert metrics["customers_targeted"] == 3
    assert metrics["customers_blanket"] == 4
    assert metrics["cost_targeted"] == 30.0

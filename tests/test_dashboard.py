"""Unit tests for dashboard helpers."""

import pandas as pd

from dashboard.filters import customer_selector_options, filter_dashboard_frame
from dashboard.metrics import compute_roi_metrics
from src.constants import SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS, SEGMENT_SURE_THINGS


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4"],
            "segment": [
                SEGMENT_PERSUADABLES,
                SEGMENT_PERSUADABLES,
                SEGMENT_SLEEPING_DOGS,
                SEGMENT_SURE_THINGS,
            ],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "churn_probability": [0.6, 0.2, 0.1, 0.7],
            "uplift": [0.2, 0.15, -0.03, 0.05],
            "expected_churn_if_treated": [0.4, 0.05, 0.13, 0.65],
            "MonthlyCharges": [70.0, 50.0, 40.0, 90.0],
        }
    )


def test_filter_dashboard_frame_by_segment_and_contract():
    df = _sample_frame()
    filtered = filter_dashboard_frame(
        df, segment_filter=SEGMENT_PERSUADABLES, contract_filter="Month-to-month"
    )
    assert len(filtered) == 1
    assert set(filtered["segment"]) == {SEGMENT_PERSUADABLES}


def test_customer_selector_options_use_index_values_and_no_nan_labels():
    df = _sample_frame()
    options = customer_selector_options(df.iloc[[2]])
    assert options[0]["value"] == 2
    assert "nan" not in options[0]["label"].lower()
    assert "A3" in options[0]["label"]
    assert "Sleeping Dogs" in options[0]["label"]


def test_compute_roi_metrics_excludes_sleeping_dogs_and_adds_business_keys():
    df = _sample_frame()
    metrics = compute_roi_metrics(df, discount_cost_per_customer=10.0)
    assert metrics["customers_targeted"] == 2
    assert metrics["customers_blanket"] == 4
    assert metrics["cost_targeted"] == 20.0
    assert "expected_revenue_saved_targeted" in metrics
    assert "campaign_cost_targeted" in metrics
    assert "net_profit_targeted" in metrics
    assert "roi_pct_targeted" in metrics
    assert "expected_customers_retained_targeted" in metrics
    assert "cost_per_retained_customer_targeted" in metrics
    # Legacy key still present
    assert "roi_targeted" in metrics

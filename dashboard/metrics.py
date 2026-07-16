"""Dashboard KPI and ROI metric helpers."""

from __future__ import annotations

import pandas as pd
from dash import html

from src.config import campaign as campaign_config
from src.constants import TARGET_SEGMENTS


def create_kpi_card(label: str, value: str, color: str = "#3498db") -> html.Div:
    """Create a styled KPI card."""

    return html.Div(
        [
            html.H4(label, style={"color": "#666", "fontSize": "14px", "marginBottom": "8px"}),
            html.H2(value, style={"color": color, "fontSize": "32px", "marginBottom": "0"}),
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "20px",
            "borderRadius": "8px",
            "border": f"2px solid {color}",
            "textAlign": "center",
        },
    )


def compute_roi_metrics(
    df: pd.DataFrame,
    discount_cost_per_customer: float | None = None,
) -> dict[str, float]:
    """Compute retention-efficiency metrics for targeted vs. blanket spend.

    Target group: Persuadables + Low-Risk Upside (excludes Sleeping Dogs).
    """

    if discount_cost_per_customer is None:
        discount_cost_per_customer = campaign_config.default_discount_cost

    targeted = df[df["segment"].isin(TARGET_SEGMENTS)]

    churn_prevented_targeted = (
        targeted["churn_probability"].sum() - targeted["expected_churn_if_treated"].sum()
    )
    cost_targeted = len(targeted) * discount_cost_per_customer

    churn_prevented_blanket = (
        df["churn_probability"].sum() - df["expected_churn_if_treated"].sum()
    )
    cost_blanket = len(df) * discount_cost_per_customer

    roi_targeted = churn_prevented_targeted / cost_targeted if cost_targeted > 0 else 0.0
    roi_blanket = churn_prevented_blanket / cost_blanket if cost_blanket > 0 else 0.0
    roi_improvement = (roi_targeted - roi_blanket) / roi_blanket if roi_blanket > 0 else 0.0

    def _retained_revenue(subset: pd.DataFrame) -> float:
        if "MonthlyCharges" not in subset.columns or len(subset) == 0:
            return 0.0
        prevented = subset["churn_probability"] - subset["expected_churn_if_treated"]
        return float((prevented * subset["MonthlyCharges"]).sum())

    return {
        "churn_prevented_targeted": float(churn_prevented_targeted),
        "cost_targeted": float(cost_targeted),
        "roi_targeted": float(roi_targeted),
        "customers_targeted": len(targeted),
        "retained_revenue_targeted": _retained_revenue(targeted),
        "churn_prevented_blanket": float(churn_prevented_blanket),
        "cost_blanket": float(cost_blanket),
        "roi_blanket": float(roi_blanket),
        "customers_blanket": len(df),
        "retained_revenue_blanket": _retained_revenue(df),
        "roi_improvement": float(roi_improvement),
    }

"""Dashboard KPI and ROI metric helpers."""

from __future__ import annotations

import pandas as pd
from dash import html

from src.config import campaign as campaign_config
from src.constants import TARGET_SEGMENTS
from src.validation import treated_churn_probability


def create_kpi_card(label: str, value: str, color: str = "#3498db") -> html.Div:
    """Create a styled KPI card."""

    return html.Div(
        [
            html.H4(label, style={"color": "#666", "fontSize": "14px", "marginBottom": "8px"}),
            html.H2(value, style={"color": color, "fontSize": "28px", "marginBottom": "0"}),
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "16px",
            "borderRadius": "8px",
            "border": f"2px solid {color}",
            "textAlign": "center",
            "minWidth": "140px",
        },
    )


def compute_roi_metrics(
    df: pd.DataFrame,
    discount_cost_per_customer: float | None = None,
    acceptance_probability: float | None = None,
) -> dict[str, float]:
    """Compute retention-efficiency and business ROI metrics.

    Backward-compatible keys (``roi_targeted``, ``cost_targeted``, …) are preserved.
    Newer business-friendly keys are also returned:

    - expected_revenue_saved_*
    - campaign_cost_*
    - net_profit_*
    - roi_pct_*
    - expected_customers_retained_*
    - cost_per_retained_customer_*
    """

    if discount_cost_per_customer is None:
        discount_cost_per_customer = campaign_config.default_discount_cost

    if acceptance_probability is None:
        acceptance_probability = getattr(campaign_config, "default_acceptance_probability", 0.60)

    # Ensure treated probabilities are valid before aggregating.
    working = df.copy()
    working["expected_churn_if_treated"] = treated_churn_probability(
        working["churn_probability"], working["uplift"]
    )

    targeted = working[working["segment"].isin(TARGET_SEGMENTS)]

    churn_prevented_targeted = float(
        (targeted["churn_probability"] - targeted["expected_churn_if_treated"]).sum()
    )
    churn_prevented_blanket = float(
        (working["churn_probability"] - working["expected_churn_if_treated"]).sum()
    )

    from src.causal.roi import expected_campaign_cost, expected_retained_revenue

    cost_targeted = expected_campaign_cost(
        discount_cost_per_customer, len(targeted), acceptance_probability
    )
    cost_blanket = expected_campaign_cost(
        discount_cost_per_customer, len(working), acceptance_probability
    )

    # Legacy efficiency metric: churn prevented per discount dollar.
    roi_targeted = churn_prevented_targeted / cost_targeted if cost_targeted > 0 else 0.0
    roi_blanket = churn_prevented_blanket / cost_blanket if cost_blanket > 0 else 0.0
    roi_improvement = (roi_targeted - roi_blanket) / roi_blanket if roi_blanket > 0 else 0.0

    revenue_targeted = expected_retained_revenue(
        monthly_charges=targeted["MonthlyCharges"] if "MonthlyCharges" in targeted.columns else pd.Series(dtype=float),
        baseline_churn=targeted["churn_probability"],
        treatment_effect=targeted["uplift"],
        acceptance_probability=acceptance_probability,
        annual_revenue_multiplier=campaign_config.annual_revenue_multiplier,
    )
    revenue_blanket = expected_retained_revenue(
        monthly_charges=working["MonthlyCharges"] if "MonthlyCharges" in working.columns else pd.Series(dtype=float),
        baseline_churn=working["churn_probability"],
        treatment_effect=working["uplift"],
        acceptance_probability=acceptance_probability,
        annual_revenue_multiplier=campaign_config.annual_revenue_multiplier,
    )

    net_profit_targeted = revenue_targeted - cost_targeted
    net_profit_blanket = revenue_blanket - cost_blanket
    roi_pct_targeted = (net_profit_targeted / cost_targeted * 100.0) if cost_targeted > 0 else 0.0
    roi_pct_blanket = (net_profit_blanket / cost_blanket * 100.0) if cost_blanket > 0 else 0.0

    retained_targeted = max(churn_prevented_targeted, 0.0)
    retained_blanket = max(churn_prevented_blanket, 0.0)
    cost_per_retained_targeted = (
        cost_targeted / retained_targeted if retained_targeted > 0 else float("inf")
    )
    cost_per_retained_blanket = (
        cost_blanket / retained_blanket if retained_blanket > 0 else float("inf")
    )

    return {
        # Legacy keys (backward compatible)
        "churn_prevented_targeted": churn_prevented_targeted,
        "cost_targeted": cost_targeted,
        "roi_targeted": float(roi_targeted),
        "customers_targeted": float(len(targeted)),
        "retained_revenue_targeted": float(
            revenue_targeted / campaign_config.annual_revenue_multiplier
        ),
        "churn_prevented_blanket": churn_prevented_blanket,
        "cost_blanket": cost_blanket,
        "roi_blanket": float(roi_blanket),
        "customers_blanket": float(len(working)),
        "retained_revenue_blanket": float(
            revenue_blanket / campaign_config.annual_revenue_multiplier
        ),
        "roi_improvement": float(roi_improvement),
        # Business-friendly keys
        "expected_revenue_saved_targeted": float(revenue_targeted),
        "expected_revenue_saved_blanket": float(revenue_blanket),
        "campaign_cost_targeted": cost_targeted,
        "campaign_cost_blanket": cost_blanket,
        "net_profit_targeted": float(net_profit_targeted),
        "net_profit_blanket": float(net_profit_blanket),
        "roi_pct_targeted": float(roi_pct_targeted),
        "roi_pct_blanket": float(roi_pct_blanket),
        "expected_customers_retained_targeted": float(retained_targeted),
        "expected_customers_retained_blanket": float(retained_blanket),
        "cost_per_retained_customer_targeted": float(cost_per_retained_targeted),
        "cost_per_retained_customer_blanket": float(cost_per_retained_blanket),
    }

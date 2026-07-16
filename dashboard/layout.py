"""Static Dash layout for the retention decision dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import dcc, html

from dashboard.metrics import create_kpi_card
from src.constants import ALL_SEGMENTS, SEGMENT_SLEEPING_DOGS


def _fmt_money(value: float) -> str:
    if value == float("inf"):
        return "n/a"
    return f"${value:,.0f}"


def _fmt_pct(value: float) -> str:
    return f"{value:,.1f}%"


def build_layout(
    df: pd.DataFrame,
    roi_metrics: dict[str, float],
    uplift_meta: dict[str, Any],
) -> html.Div:
    """Construct the full dashboard layout."""

    segment_options = [{"label": "All Segments", "value": "all"}] + [
        {
            "label": (
                f"{seg} (do not target)" if seg == SEGMENT_SLEEPING_DOGS else seg
            ),
            "value": seg,
        }
        for seg in ALL_SEGMENTS
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Causal Churn Intervention System",
                        style={"textAlign": "center", "marginBottom": "10px"},
                    ),
                    html.P(
                        "Decision-support tool for retention budget allocation — not a churn classifier.",
                        style={"textAlign": "center", "color": "#666", "fontSize": "16px"},
                    ),
                    html.P(
                        f"Model validity — CATE recovery correlation: {uplift_meta['cate_recovery_corr']:.2f} "
                        f"(MAE: {uplift_meta['cate_recovery_mae']:.3f})",
                        style={
                            "textAlign": "center",
                            "color": "#888",
                            "fontSize": "13px",
                            "marginTop": "6px",
                        },
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
            html.Div(
                [
                    html.H2(
                        "Campaign Economics: Targeted vs. Blanket Spend",
                        style={"marginBottom": "20px", "color": "#222"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Targeted Strategy (Persuadables)",
                                        style={"color": "#2ecc71", "marginBottom": "15px"},
                                    ),
                                    html.Div(
                                        [
                                            create_kpi_card(
                                                "Expected Revenue Saved",
                                                _fmt_money(
                                                    roi_metrics["expected_revenue_saved_targeted"]
                                                ),
                                                "#2ecc71",
                                            ),
                                            create_kpi_card(
                                                "Campaign Cost",
                                                _fmt_money(roi_metrics["campaign_cost_targeted"]),
                                                "#27ae60",
                                            ),
                                            create_kpi_card(
                                                "Net Profit",
                                                _fmt_money(roi_metrics["net_profit_targeted"]),
                                                "#1abc9c",
                                            ),
                                            create_kpi_card(
                                                "ROI %",
                                                _fmt_pct(roi_metrics["roi_pct_targeted"]),
                                                "#16a085",
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                    html.P(
                                        f"Customers targeted: {int(roi_metrics['customers_targeted'])} "
                                        f"| Expected retained: "
                                        f"{roi_metrics['expected_customers_retained_targeted']:.1f}",
                                        style={"fontSize": "13px", "color": "#666", "marginTop": "12px"},
                                    ),
                                    html.P(
                                        f"Cost per retained customer: "
                                        f"{_fmt_money(roi_metrics['cost_per_retained_customer_targeted'])}",
                                        style={"fontSize": "13px", "color": "#666"},
                                    ),
                                    html.P(
                                        f"Legacy efficiency (churn prevented / $): "
                                        f"{roi_metrics['roi_targeted']:.3f}x",
                                        style={"fontSize": "12px", "color": "#999", "marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "1", "marginRight": "20px"},
                            ),
                            html.Div(
                                [
                                    html.H3(
                                        "Blanket Strategy",
                                        style={"color": "#e74c3c", "marginBottom": "15px"},
                                    ),
                                    html.Div(
                                        [
                                            create_kpi_card(
                                                "Expected Revenue Saved",
                                                _fmt_money(
                                                    roi_metrics["expected_revenue_saved_blanket"]
                                                ),
                                                "#e74c3c",
                                            ),
                                            create_kpi_card(
                                                "Campaign Cost",
                                                _fmt_money(roi_metrics["campaign_cost_blanket"]),
                                                "#c0392b",
                                            ),
                                            create_kpi_card(
                                                "Net Profit",
                                                _fmt_money(roi_metrics["net_profit_blanket"]),
                                                "#d35400",
                                            ),
                                            create_kpi_card(
                                                "ROI %",
                                                _fmt_pct(roi_metrics["roi_pct_blanket"]),
                                                "#e67e22",
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                    html.P(
                                        f"Customers targeted: {int(roi_metrics['customers_blanket'])} "
                                        f"| Expected retained: "
                                        f"{roi_metrics['expected_customers_retained_blanket']:.1f}",
                                        style={"fontSize": "13px", "color": "#666", "marginTop": "12px"},
                                    ),
                                    html.P(
                                        f"Cost per retained customer: "
                                        f"{_fmt_money(roi_metrics['cost_per_retained_customer_blanket'])}",
                                        style={"fontSize": "13px", "color": "#666"},
                                    ),
                                    html.P(
                                        f"Legacy efficiency (churn prevented / $): "
                                        f"{roi_metrics['roi_blanket']:.3f}x",
                                        style={"fontSize": "12px", "color": "#999", "marginTop": "6px"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-around"},
                    ),
                    html.Div(
                        [
                            html.H3(
                                f"Targeted efficiency is {roi_metrics['roi_improvement']:.1%} better "
                                f"(legacy churn-prevented / $ metric)",
                                style={"color": "#2ecc71", "marginTop": "25px", "textAlign": "center"},
                            ),
                        ],
                        style={
                            "backgroundColor": "#f0fdf4",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "marginTop": "25px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#fff",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "marginBottom": "30px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.H2("Customer Risk Table", style={"marginBottom": "15px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Filter by Segment:", style={"fontWeight": "bold"}),
                                    dcc.Dropdown(
                                        id="segment-filter",
                                        options=segment_options,
                                        value="all",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"width": "30%", "display": "inline-block", "marginRight": "20px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Filter by Contract:", style={"fontWeight": "bold"}),
                                    dcc.Dropdown(
                                        id="contract-filter",
                                        options=[{"label": "All Contracts", "value": "all"}]
                                        + [
                                            {"label": str(contract), "value": contract}
                                            for contract in sorted(
                                                df["Contract"].dropna().astype(str).unique()
                                            )
                                        ],
                                        value="all",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"width": "30%", "display": "inline-block"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(id="customer-table-container", style={"overflowX": "auto"}),
                ],
                style={
                    "backgroundColor": "#fff",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "marginBottom": "30px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Segment Distribution", style={"marginBottom": "15px"}),
                            dcc.Graph(id="segment-distribution"),
                        ],
                        style={"flex": "1", "marginRight": "15px"},
                    ),
                    html.Div(
                        [
                            html.H2("Churn Risk Distribution", style={"marginBottom": "15px"}),
                            dcc.Graph(id="churn-distribution"),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "marginBottom": "30px"},
            ),
            html.Div(
                [
                    html.H2("Customer Detail View", style={"marginBottom": "15px"}),
                    html.Div(
                        [
                            html.Label("Select a customer:", style={"fontWeight": "bold"}),
                            dcc.Dropdown(id="customer-selector", style={"width": "100%"}),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        id="customer-detail",
                        style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px"},
                    ),
                ],
                style={
                    "backgroundColor": "#fff",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
        ],
        style={
            "maxWidth": "1400px",
            "margin": "0 auto",
            "padding": "30px",
            "backgroundColor": "#ecf0f1",
            "minHeight": "100vh",
            "fontFamily": "Arial, sans-serif",
        },
    )

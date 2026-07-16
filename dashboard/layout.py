"""Static Dash layout for the retention decision dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
from dash import dcc, html

from dashboard.metrics import create_kpi_card
from src.constants import ALL_SEGMENTS, SEGMENT_LOW_RISK_UPSIDE, SEGMENT_PERSUADABLES, SEGMENT_SLEEPING_DOGS


def build_layout(
    df: pd.DataFrame,
    roi_metrics: dict[str, float],
    uplift_meta: dict[str, Any],
) -> html.Div:
    """Construct the full dashboard layout."""

    segment_options = [
        {"label": "All Segments", "value": "all"},
        {"label": SEGMENT_PERSUADABLES, "value": SEGMENT_PERSUADABLES},
        {"label": SEGMENT_LOW_RISK_UPSIDE, "value": SEGMENT_LOW_RISK_UPSIDE},
        {"label": f"{SEGMENT_SLEEPING_DOGS} (do not target)", "value": SEGMENT_SLEEPING_DOGS},
    ]
    for seg in ALL_SEGMENTS:
        if seg not in {SEGMENT_PERSUADABLES, SEGMENT_LOW_RISK_UPSIDE, SEGMENT_SLEEPING_DOGS}:
            segment_options.append({"label": seg, "value": seg})

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
                        "Efficiency Comparison: Targeted vs. Blanket Spend",
                        style={"marginBottom": "20px", "color": "#222"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "Targeted Strategy",
                                        style={"color": "#2ecc71", "marginBottom": "15px"},
                                    ),
                                    create_kpi_card(
                                        "Churn Prevented per Discount Dollar",
                                        f"{roi_metrics['roi_targeted']:.2f}x",
                                        "#2ecc71",
                                    ),
                                    html.P(
                                        f"Target Persuadables + Low-Risk Upside: "
                                        f"{roi_metrics['customers_targeted']} customers",
                                        style={"fontSize": "13px", "color": "#666", "marginTop": "8px"},
                                    ),
                                    html.P(
                                        f"Cost: ${roi_metrics['cost_targeted']:.0f}",
                                        style={"fontSize": "13px", "color": "#666"},
                                    ),
                                    html.P(
                                        f"Expected retained monthly revenue: "
                                        f"${roi_metrics['retained_revenue_targeted']:.0f}",
                                        style={"fontSize": "13px", "color": "#666"},
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
                                    create_kpi_card(
                                        "Churn Prevented per Discount Dollar",
                                        f"{roi_metrics['roi_blanket']:.2f}x",
                                        "#e74c3c",
                                    ),
                                    html.P(
                                        f"Discount everyone: {roi_metrics['customers_blanket']} customers",
                                        style={"fontSize": "13px", "color": "#666", "marginTop": "8px"},
                                    ),
                                    html.P(
                                        f"Cost: ${roi_metrics['cost_blanket']:.0f}",
                                        style={"fontSize": "13px", "color": "#666"},
                                    ),
                                    html.P(
                                        f"Expected retained monthly revenue: "
                                        f"${roi_metrics['retained_revenue_blanket']:.0f}",
                                        style={"fontSize": "13px", "color": "#666"},
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
                                f"Targeted efficiency is {roi_metrics['roi_improvement']:.1%} better",
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
                                            {"label": contract, "value": contract}
                                            for contract in sorted(df["Contract"].astype(str).unique())
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

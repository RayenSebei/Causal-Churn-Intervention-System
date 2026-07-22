"""Dash callbacks for the retention decision dashboard."""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, html

from dashboard.filters import _safe_segment, customer_selector_options, filter_dashboard_frame
from dashboard.plots import churn_distribution_figure, segment_distribution_figure
from src.config import plots as plot_config
from src.constants import ALL_SEGMENTS, CUSTOMER_ID_COLUMN, SEGMENT_COLORS
from src.validation import summarize_segment_distribution, treated_churn_probability


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Attach all interactive callbacks to the Dash app."""

    # Ensure treated probabilities stay valid even if the frame was mutated.
    df = df.copy()
    df["expected_churn_if_treated"] = treated_churn_probability(
        df["churn_probability"], df["uplift"]
    )

    # Real consistency check: compare how the chart builds its counts
    # (value_counts with dropna=False, matching plots.py) against the
    # canonical summarize_segment_distribution helper.
    chart_value_counts = df["segment"].value_counts(dropna=False)
    chart_counts: dict[str, int] = {
        seg: int(chart_value_counts.get(seg, 0)) for seg in ALL_SEGMENTS
    }
    # Include any non-canonical keys (e.g. NaN) so the check catches them.
    for key, count in chart_value_counts.items():
        str_key = str(key)
        if str_key not in chart_counts:
            chart_counts[str_key] = int(count)
    chart_counts["total"] = int(chart_value_counts.sum())
    table_counts = summarize_segment_distribution(df["segment"])
    if chart_counts != table_counts:
        raise RuntimeError(
            f"Segment count inconsistency at callback registration: "
            f"chart={chart_counts}, table={table_counts}"
        )

    @app.callback(
        Output("customer-table-container", "children"),
        Input("segment-filter", "value"),
        Input("contract-filter", "value"),
    )
    def update_customer_table(segment_filter, contract_filter):
        filtered_df = filter_dashboard_frame(
            df, segment_filter=segment_filter, contract_filter=contract_filter
        )

        cols = [
            c
            for c in [
                CUSTOMER_ID_COLUMN,
                "churn_probability",
                "uplift",
                "expected_churn_if_treated",
                "segment",
                "Contract",
                "tenure",
                "MonthlyCharges",
            ]
            if c in filtered_df.columns
        ]
        display_df = filtered_df[cols].copy()
        rename = {
            CUSTOMER_ID_COLUMN: "Customer ID",
            "churn_probability": "Churn Prob",
            "uplift": "Uplift",
            "expected_churn_if_treated": "Expected Churn (Treated)",
            "segment": "Segment",
            "Contract": "Contract",
            "tenure": "Tenure",
            "MonthlyCharges": "Monthly $",
        }
        display_df = display_df.rename(columns=rename)
        if "Churn Prob" in display_df.columns:
            display_df["Churn Prob"] = display_df["Churn Prob"].apply(lambda x: f"{float(x):.1%}")
        if "Uplift" in display_df.columns:
            display_df["Uplift"] = display_df["Uplift"].apply(lambda x: f"{float(x):+.1%}")
        if "Expected Churn (Treated)" in display_df.columns:
            display_df["Expected Churn (Treated)"] = display_df["Expected Churn (Treated)"].apply(
                lambda x: f"{float(x):.1%}"
            )
        if "Tenure" in display_df.columns:
            display_df["Tenure"] = display_df["Tenure"].astype(int)
        if "Monthly $" in display_df.columns:
            display_df["Monthly $"] = display_df["Monthly $"].apply(lambda x: f"${float(x):.0f}")
        if "Segment" in display_df.columns:
            display_df["Segment"] = display_df["Segment"].apply(_safe_segment)

        max_rows = min(plot_config.max_table_rows, len(display_df))
        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(
                                col,
                                style={
                                    "padding": "10px",
                                    "textAlign": "left",
                                    "backgroundColor": "#ddd",
                                    "fontWeight": "bold",
                                },
                            )
                            for col in display_df.columns
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    str(display_df.iloc[i][col]),
                                    style={
                                        "padding": "10px",
                                        "borderBottom": "1px solid #ddd",
                                        "backgroundColor": (
                                            SEGMENT_COLORS.get(
                                                _safe_segment(filtered_df.iloc[i]["segment"]), "#fff"
                                            )
                                            if col == "Segment"
                                            else "#fff"
                                        ),
                                    },
                                )
                                for col in display_df.columns
                            ]
                        )
                        for i in range(max_rows)
                    ]
                ),
            ],
            style={"width": "100%", "borderCollapse": "collapse"},
        )

    @app.callback(
        Output("segment-distribution", "figure"),
        Input("segment-filter", "value"),
    )
    def update_segment_chart(_segment_filter):
        return segment_distribution_figure(df)

    @app.callback(
        Output("churn-distribution", "figure"),
        Input("contract-filter", "value"),
    )
    def update_churn_chart(contract_filter):
        chart_df = filter_dashboard_frame(df, contract_filter=contract_filter)
        return churn_distribution_figure(chart_df)

    @app.callback(
        Output("customer-selector", "options"),
        Input("segment-filter", "value"),
    )
    def update_customer_selector(segment_filter):
        selector_df = filter_dashboard_frame(df, segment_filter=segment_filter)
        return customer_selector_options(selector_df)

    @app.callback(
        Output("customer-detail", "children"),
        Input("customer-selector", "value"),
    )
    def update_customer_detail(customer_idx):
        if customer_idx is None:
            return html.Div("Select a customer to view details.")

        customer = df.loc[customer_idx]
        customer_id = (
            str(customer[CUSTOMER_ID_COLUMN])
            if CUSTOMER_ID_COLUMN in customer.index and pd.notna(customer[CUSTOMER_ID_COLUMN])
            else f"CUST-{customer_idx}"
        )
        segment = _safe_segment(customer.get("segment"))
        explanation = customer.get("shap_explanation", "No explanation available")
        if pd.isna(explanation) or str(explanation).strip() == "":
            explanation = "No explanation available"

        return html.Div(
            [
                html.H3(f"Customer ID: {customer_id}", style={"marginBottom": "15px"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    f"Baseline Churn Prob: {float(customer['churn_probability']):.2%}",
                                    style={"fontSize": "16px"},
                                ),
                                html.P(
                                    f"Segment: {segment}",
                                    style={
                                        "fontSize": "16px",
                                        "color": SEGMENT_COLORS.get(segment, "#000"),
                                    },
                                ),
                                html.P(
                                    f"Estimated Treatment Effect: {float(customer['uplift']):+.2%}",
                                    style={"fontSize": "16px"},
                                ),
                                html.P(
                                    f"Expected Churn if Treated: "
                                    f"{float(customer['expected_churn_if_treated']):.2%}",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Contract: {customer['Contract']}",
                                    style={"fontSize": "14px"},
                                ),
                                html.P(
                                    f"Tenure: {int(customer['tenure'])} months",
                                    style={"fontSize": "14px"},
                                ),
                                html.P(
                                    f"Monthly Charges: ${float(customer['MonthlyCharges']):.2f}",
                                    style={"fontSize": "14px"},
                                ),
                                html.P(
                                    f"Total Charges: ${float(customer['TotalCharges']):.2f}",
                                    style={"fontSize": "14px"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                    ],
                    style={"display": "flex", "marginBottom": "20px"},
                ),
                html.H4("Why High Churn Risk?", style={"marginBottom": "10px", "color": "#e74c3c"}),
                html.P(
                    str(explanation),
                    style={"fontSize": "14px", "lineHeight": "1.6", "color": "#333"},
                ),
            ]
        )

"""Dash callbacks for the retention decision dashboard."""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, html

from dashboard.filters import customer_selector_options, filter_dashboard_frame
from dashboard.plots import churn_distribution_figure, segment_distribution_figure
from src.config import plots as plot_config
from src.constants import SEGMENT_COLORS


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Attach all interactive callbacks to the Dash app."""

    @app.callback(
        Output("customer-table-container", "children"),
        Input("segment-filter", "value"),
        Input("contract-filter", "value"),
    )
    def update_customer_table(segment_filter, contract_filter):
        filtered_df = filter_dashboard_frame(
            df, segment_filter=segment_filter, contract_filter=contract_filter
        )

        display_df = filtered_df[
            [
                "churn_probability",
                "uplift",
                "expected_churn_if_treated",
                "segment",
                "Contract",
                "tenure",
                "MonthlyCharges",
            ]
        ].copy()
        display_df.columns = [
            "Churn Prob",
            "Uplift",
            "Expected Churn (Treated)",
            "Segment",
            "Contract",
            "Tenure",
            "Monthly $",
        ]
        display_df["Churn Prob"] = display_df["Churn Prob"].apply(lambda x: f"{x:.1%}")
        display_df["Uplift"] = display_df["Uplift"].apply(lambda x: f"{x:+.1%}")
        display_df["Expected Churn (Treated)"] = display_df["Expected Churn (Treated)"].apply(
            lambda x: f"{x:.1%}"
        )
        display_df["Tenure"] = display_df["Tenure"].astype(int)
        display_df["Monthly $"] = display_df["Monthly $"].apply(lambda x: f"${x:.0f}")

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
                                            SEGMENT_COLORS.get(filtered_df.iloc[i]["segment"], "#fff")
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
        return html.Div(
            [
                html.H3(f"Customer Index: {customer_idx}", style={"marginBottom": "15px"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    f"Baseline Churn Prob: {customer['churn_probability']:.2%}",
                                    style={"fontSize": "16px"},
                                ),
                                html.P(
                                    f"Segment: {customer['segment']}",
                                    style={
                                        "fontSize": "16px",
                                        "color": SEGMENT_COLORS.get(customer["segment"], "#000"),
                                    },
                                ),
                                html.P(
                                    f"Estimated Uplift: {customer['uplift']:+.2%}",
                                    style={"fontSize": "16px"},
                                ),
                                html.P(
                                    f"Expected Churn if Treated: {customer['expected_churn_if_treated']:.2%}",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.P(f"Contract: {customer['Contract']}", style={"fontSize": "14px"}),
                                html.P(
                                    f"Tenure: {customer['tenure']} months",
                                    style={"fontSize": "14px"},
                                ),
                                html.P(
                                    f"Monthly Charges: ${customer['MonthlyCharges']:.2f}",
                                    style={"fontSize": "14px"},
                                ),
                                html.P(
                                    f"Total Charges: ${customer['TotalCharges']:.2f}",
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
                    customer["shap_explanation"],
                    style={"fontSize": "14px", "lineHeight": "1.6", "color": "#333"},
                ),
            ]
        )

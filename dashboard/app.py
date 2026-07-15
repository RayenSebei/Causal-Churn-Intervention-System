"""Dash-based decision-support dashboard for retention budget allocation."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, Input, Output, callback
import numpy as np

from src.dashboard_data import load_dashboard_data, compute_roi_metrics

DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "baseline_churn_model.joblib"

dashboard_data = load_dashboard_data(DATA_PATH, MODEL_PATH)
df = dashboard_data["df"].reset_index(drop=True)
roi_metrics = compute_roi_metrics(df)
uplift_meta = dashboard_data["uplift_results"]

app = dash.Dash(__name__)

SEGMENT_COLORS = {
    "Persuadables": "#2ecc71",
    "Sure Things": "#3498db",
    "Lost Causes": "#e74c3c",
    "Sleeping Dogs": "#f39c12",
    "Low-Risk Upside": "#9b59b6",
}

def create_kpi_card(label: str, value: str, color: str = "#3498db"):
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
        }
    )

app.layout = html.Div(
    [
        html.Div([
            html.H1("Causal Churn Intervention System", style={"textAlign": "center", "marginBottom": "10px"}),
            html.P(
                "Decision-support tool for retention budget allocation — not a churn classifier.",
                style={"textAlign": "center", "color": "#666", "fontSize": "16px"}
            ),
            html.P(
                f"Model validity — CATE recovery correlation: {uplift_meta['cate_recovery_corr']:.2f} "
                f"(MAE: {uplift_meta['cate_recovery_mae']:.3f})",
                style={"textAlign": "center", "color": "#888", "fontSize": "13px", "marginTop": "6px"}
            ),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.H2("Efficiency Comparison: Targeted vs. Blanket Spend", style={"marginBottom": "20px", "color": "#222"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Targeted Strategy", style={"color": "#2ecc71", "marginBottom": "15px"}),
                            create_kpi_card(
                                "Churn Prevented per Discount Dollar",
                                f"{roi_metrics['roi_targeted']:.2f}x",
                                "#2ecc71"
                            ),
                            html.Div(style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                            html.P(
                                f"Target Persuadables + Low-Risk Upside: {roi_metrics['customers_targeted']} customers",
                                style={"fontSize": "13px", "color": "#666", "marginTop": "8px"}
                            ),
                            html.P(
                                f"Cost: ${roi_metrics['cost_targeted']:.0f}",
                                style={"fontSize": "13px", "color": "#666"}
                            ),
                            html.P(
                                f"Expected retained monthly revenue: ${roi_metrics['retained_revenue_targeted']:.0f}",
                                style={"fontSize": "13px", "color": "#666"}
                            ),
                        ],
                        style={"flex": "1", "marginRight": "20px"}
                    ),
                    html.Div(
                        [
                            html.H3("Blanket Strategy", style={"color": "#e74c3c", "marginBottom": "15px"}),
                            create_kpi_card(
                                "Churn Prevented per Discount Dollar",
                                f"{roi_metrics['roi_blanket']:.2f}x",
                                "#e74c3c"
                            ),
                            html.Div(style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                            html.P(
                                f"Discount everyone: {roi_metrics['customers_blanket']} customers",
                                style={"fontSize": "13px", "color": "#666", "marginTop": "8px"}
                            ),
                            html.P(
                                f"Cost: ${roi_metrics['cost_blanket']:.0f}",
                                style={"fontSize": "13px", "color": "#666"}
                            ),
                            html.P(
                                f"Expected retained monthly revenue: ${roi_metrics['retained_revenue_blanket']:.0f}",
                                style={"fontSize": "13px", "color": "#666"}
                            ),
                        ],
                        style={"flex": "1"}
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-around"}
            ),
            html.Div(
                [
                    html.H3(
                        f"Targeted efficiency is {roi_metrics['roi_improvement']:.1%} better",
                        style={"color": "#2ecc71", "marginTop": "25px", "textAlign": "center"}
                    ),
                ],
                style={"backgroundColor": "#f0fdf4", "padding": "15px", "borderRadius": "8px", "marginTop": "25px"}
            ),
        ], style={"backgroundColor": "#fff", "padding": "25px", "borderRadius": "10px", "marginBottom": "30px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H2("Customer Risk Table", style={"marginBottom": "15px"}),
            html.Div([
                html.Div([
                    html.Label("Filter by Segment:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="segment-filter",
                        options=[
                            {"label": "All Segments", "value": "all"},
                            {"label": "Persuadables", "value": "Persuadables"},
                            {"label": "Low-Risk Upside", "value": "Low-Risk Upside"},
                            {"label": "Sleeping Dogs (do not target)", "value": "Sleeping Dogs"},
                            {"label": "Sure Things", "value": "Sure Things"},
                            {"label": "Lost Causes", "value": "Lost Causes"},
                        ],
                        value="all",
                        style={"width": "100%"}
                    ),
                ], style={"width": "30%", "display": "inline-block", "marginRight": "20px"}),
                html.Div([
                    html.Label("Filter by Contract:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="contract-filter",
                        options=[
                            {"label": "All Contracts", "value": "all"},
                        ] + [
                            {"label": contract, "value": contract}
                            for contract in df["Contract"].unique()
                        ],
                        value="all",
                        style={"width": "100%"}
                    ),
                ], style={"width": "30%", "display": "inline-block"}),
            ], style={"marginBottom": "20px"}),

            html.Div(id="customer-table-container", style={"overflowX": "auto"}),
        ], style={"backgroundColor": "#fff", "padding": "25px", "borderRadius": "10px", "marginBottom": "30px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.Div([
                html.H2("Segment Distribution", style={"marginBottom": "15px"}),
                dcc.Graph(id="segment-distribution"),
            ], style={"flex": "1", "marginRight": "15px"}),
            html.Div([
                html.H2("Churn Risk Distribution", style={"marginBottom": "15px"}),
                dcc.Graph(id="churn-distribution"),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "30px"}),

        html.Div([
            html.H2("Customer Detail View", style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Select a customer:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="customer-selector", style={"width": "100%"}),
            ], style={"marginBottom": "20px"}),
            html.Div(id="customer-detail", style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px"}),
        ], style={"backgroundColor": "#fff", "padding": "25px", "borderRadius": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "30px", "backgroundColor": "#ecf0f1", "minHeight": "100vh", "fontFamily": "Arial, sans-serif"}
)


@callback(
    Output("customer-table-container", "children"),
    Input("segment-filter", "value"),
    Input("contract-filter", "value"),
)
def update_customer_table(segment_filter, contract_filter):
    """Update customer risk table based on filters."""
    filtered_df = df.copy()

    if segment_filter != "all":
        filtered_df = filtered_df[filtered_df["segment"] == segment_filter]

    if contract_filter != "all":
        filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]

    display_df = filtered_df[
        ["churn_probability", "uplift", "expected_churn_if_treated", "segment", "Contract", "tenure", "MonthlyCharges"]
    ].copy()
    display_df.columns = ["Churn Prob", "Uplift", "Expected Churn (Treated)", "Segment", "Contract", "Tenure", "Monthly $"]
    display_df["Churn Prob"] = display_df["Churn Prob"].apply(lambda x: f"{x:.1%}")
    display_df["Uplift"] = display_df["Uplift"].apply(lambda x: f"{x:+.1%}")
    display_df["Expected Churn (Treated)"] = display_df["Expected Churn (Treated)"].apply(lambda x: f"{x:.1%}")
    display_df["Tenure"] = display_df["Tenure"].astype(int)
    display_df["Monthly $"] = display_df["Monthly $"].apply(lambda x: f"${x:.0f}")

    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, style={"padding": "10px", "textAlign": "left", "backgroundColor": "#ddd", "fontWeight": "bold"}) for col in display_df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(str(display_df.iloc[i][col]), style={
                    "padding": "10px",
                    "borderBottom": "1px solid #ddd",
                    "backgroundColor": SEGMENT_COLORS.get(filtered_df.iloc[i]["segment"], "#fff") if col == "Segment" else "#fff"
                })
                for col in display_df.columns
            ])
            for i in range(min(50, len(display_df)))
        ]),
    ], style={"width": "100%", "borderCollapse": "collapse"})


@callback(
    Output("segment-distribution", "figure"),
    Input("segment-filter", "value"),
)
def update_segment_chart(segment_filter):
    """Update segment distribution chart."""
    segment_counts = df["segment"].value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=segment_counts.index,
            y=segment_counts.values,
            marker=dict(color=[SEGMENT_COLORS.get(seg, "#95a5a6") for seg in segment_counts.index]),
            text=segment_counts.values,
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Customer Segments",
        xaxis_title="Segment",
        yaxis_title="Count",
        hovermode="x unified",
    )
    return fig


@callback(
    Output("churn-distribution", "figure"),
    Input("contract-filter", "value"),
)
def update_churn_chart(contract_filter):
    """Update churn probability distribution."""
    if contract_filter != "all":
        chart_df = df[df["Contract"] == contract_filter]
    else:
        chart_df = df

    fig = px.histogram(chart_df, x="churn_probability", nbins=30, title="Baseline Churn Probability Distribution")
    fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Count", hovermode="x unified")
    return fig


@callback(
    Output("customer-selector", "options"),
    Input("segment-filter", "value"),
)
def update_customer_selector(segment_filter):
    """Update customer selector options using full-df index labels (not filtered positions)."""
    if segment_filter != "all":
        selector_df = df[df["segment"] == segment_filter]
    else:
        selector_df = df

    options = [
        {
            "label": (
                f"Customer {idx}: {row.segment} "
                f"({row.churn_probability:.1%} churn, {row.uplift:+.1%} uplift)"
            ),
            "value": idx,
        }
        for idx, row in selector_df.iterrows()
    ]
    return options


@callback(
    Output("customer-detail", "children"),
    Input("customer-selector", "value"),
)
def update_customer_detail(customer_idx):
    """Update customer detail view with SHAP explanation."""
    if customer_idx is None:
        return html.Div("Select a customer to view details.")

    customer = df.loc[customer_idx]
    return html.Div([
        html.H3(f"Customer Index: {customer_idx}", style={"marginBottom": "15px"}),
        html.Div([
            html.Div([
                html.P(f"Baseline Churn Prob: {customer['churn_probability']:.2%}", style={"fontSize": "16px"}),
                html.P(f"Segment: {customer['segment']}", style={"fontSize": "16px", "color": SEGMENT_COLORS.get(customer['segment'], "#000")}),
                html.P(f"Estimated Uplift: {customer['uplift']:+.2%}", style={"fontSize": "16px"}),
                html.P(f"Expected Churn if Treated: {customer['expected_churn_if_treated']:.2%}", style={"fontSize": "16px"}),
            ], style={"flex": "1"}),
            html.Div([
                html.P(f"Contract: {customer['Contract']}", style={"fontSize": "14px"}),
                html.P(f"Tenure: {customer['tenure']} months", style={"fontSize": "14px"}),
                html.P(f"Monthly Charges: ${customer['MonthlyCharges']:.2f}", style={"fontSize": "14px"}),
                html.P(f"Total Charges: ${customer['TotalCharges']:.2f}", style={"fontSize": "14px"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "20px"}),
        html.H4("Why High Churn Risk?", style={"marginBottom": "10px", "color": "#e74c3c"}),
        html.P(customer['shap_explanation'], style={"fontSize": "14px", "lineHeight": "1.6", "color": "#333"}),
    ])


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

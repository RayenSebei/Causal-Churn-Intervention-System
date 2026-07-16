"""Plot-generation helpers for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEGMENT_COLORS = {
    "Persuadables": "#2ecc71",
    "Sure Things": "#3498db",
    "Lost Causes": "#e74c3c",
    "Sleeping Dogs": "#f39c12",
    "Low-Risk Upside": "#9b59b6",
}


def segment_distribution_figure(df: pd.DataFrame) -> go.Figure:
    """Build the segment bar chart."""

    segment_counts = df["segment"].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=segment_counts.index,
                y=segment_counts.values,
                marker=dict(color=[SEGMENT_COLORS.get(seg, "#95a5a6") for seg in segment_counts.index]),
                text=segment_counts.values,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(title="Customer Segments", xaxis_title="Segment", yaxis_title="Count", hovermode="x unified")
    return fig


def churn_distribution_figure(df: pd.DataFrame) -> go.Figure:
    """Build the churn probability histogram."""

    fig = px.histogram(df, x="churn_probability", nbins=30, title="Baseline Churn Probability Distribution")
    fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Count", hovermode="x unified")
    return fig

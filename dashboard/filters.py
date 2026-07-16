"""Dashboard filter helpers."""

from __future__ import annotations

import pandas as pd


def filter_dashboard_frame(
    df: pd.DataFrame,
    *,
    segment_filter: str = "all",
    contract_filter: str = "all",
) -> pd.DataFrame:
    """Return a filtered dashboard dataframe."""

    filtered = df.copy()
    if segment_filter != "all":
        filtered = filtered.loc[filtered["segment"] == segment_filter]
    if contract_filter != "all":
        filtered = filtered.loc[filtered["Contract"] == contract_filter]
    return filtered


def customer_selector_options(df: pd.DataFrame) -> list[dict[str, object]]:
    """Build dropdown options keyed by the full dataframe index."""

    return [
        {
            "label": (
                f"Customer {idx}: {row.segment} "
                f"({row.churn_probability:.1%} churn, {row.uplift:+.1%} uplift)"
            ),
            "value": idx,
        }
        for idx, row in df.iterrows()
    ]

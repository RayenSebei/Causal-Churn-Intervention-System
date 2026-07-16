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

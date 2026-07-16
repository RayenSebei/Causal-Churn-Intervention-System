"""Dashboard filter helpers."""

from __future__ import annotations

import pandas as pd

from src.constants import CUSTOMER_ID_COLUMN


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


def _safe_segment(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return "Unknown" if text.lower() in {"", "nan", "none"} else text


def _safe_customer_id(row: pd.Series, idx: object) -> str:
    if CUSTOMER_ID_COLUMN in row.index and pd.notna(row[CUSTOMER_ID_COLUMN]):
        text = str(row[CUSTOMER_ID_COLUMN]).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return f"CUST-{idx}"


def customer_selector_options(df: pd.DataFrame) -> list[dict[str, object]]:
    """Build dropdown options with ID, segment, churn, and treatment effect."""

    options: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        customer_id = _safe_customer_id(row, idx)
        segment = _safe_segment(row.get("segment"))
        churn = float(row["churn_probability"]) if pd.notna(row["churn_probability"]) else 0.0
        uplift = float(row["uplift"]) if pd.notna(row["uplift"]) else 0.0
        options.append(
            {
                "label": (
                    f"ID {customer_id} | {segment} | "
                    f"Churn {churn:.1%} | Effect {uplift:+.1%}"
                ),
                "value": idx,
            }
        )
    return options

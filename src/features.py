"""Feature engineering helpers for the Telco churn project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SERVICE_FLAG_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create tenure buckets, service counts, and charge-to-tenure ratio."""

    featured = df.copy()

    featured["tenure_bucket"] = pd.cut(
        featured["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )

    service_flags = pd.DataFrame(index=featured.index)
    service_flags["PhoneService"] = featured["PhoneService"].astype(str).eq("Yes").astype(int)
    service_flags["MultipleLines"] = featured["MultipleLines"].astype(str).eq("Yes").astype(int)
    service_flags["InternetService"] = featured["InternetService"].astype(str).ne("No").astype(int)
    for column in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        service_flags[column] = featured[column].astype(str).eq("Yes").astype(int)

    featured["total_service_count"] = service_flags.sum(axis=1)

    featured["charge_to_tenure_ratio"] = featured["TotalCharges"] / featured["tenure"].replace({0: pd.NA})
    return featured


def save_featured_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
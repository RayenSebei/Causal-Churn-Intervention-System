"""Feature engineering for the Telco churn dataset.

Creates derived features on top of the cleaned dataframe:
- Tenure buckets
- Total service count
- Charge-to-tenure ratio
- Average monthly charge (TotalCharges / tenure)
- Contract risk score (ordinal encoding of contract length)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import paths as path_config
from src.constants import (
    ALL_SERVICE_COLUMNS,
    TENURE_BIN_EDGES,
    TENURE_BIN_LABELS,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# Kept for backward compatibility — old code may reference this.
SERVICE_FLAG_COLUMNS = ALL_SERVICE_COLUMNS


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_service_count(df: pd.DataFrame) -> pd.Series:
    """Count active services per customer (vectorised).

    A service is "active" if its value is ``"Yes"`` (for flag columns) or
    not ``"No"`` (for InternetService).

    Args:
        df: DataFrame with raw service columns.

    Returns:
        Integer series with per-row service counts.
    """
    flags = pd.DataFrame(index=df.index)

    if "PhoneService" in df.columns:
        flags["PhoneService"] = df["PhoneService"].astype(str).eq("1") | df["PhoneService"].astype(str).eq("Yes")
    if "InternetService" in df.columns:
        flags["InternetService"] = ~df["InternetService"].astype(str).isin(["No", "0"])
    if "MultipleLines" in df.columns:
        flags["MultipleLines"] = df["MultipleLines"].astype(str).eq("Yes")

    detail_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in detail_cols:
        if col in df.columns:
            flags[col] = df[col].astype(str).eq("Yes")

    return flags.sum(axis=1).astype(int)


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered feature columns on a cleaned dataframe.

    Added columns:
        - ``tenure_bucket``: Categorical tenure range.
        - ``total_service_count``: Number of active services.
        - ``charge_to_tenure_ratio``: TotalCharges / tenure.
        - ``avg_monthly_charge``: TotalCharges / tenure (alias, safe div).
        - ``contract_risk_score``: Ordinal risk encoding of Contract type.

    Args:
        df: Cleaned dataframe from ``preprocessing.clean_telco_data``.

    Returns:
        Copy of *df* with new feature columns appended.
    """
    featured = df.copy()

    # Tenure buckets
    featured["tenure_bucket"] = pd.cut(
        featured["tenure"],
        bins=TENURE_BIN_EDGES,
        labels=TENURE_BIN_LABELS,
    )

    # Service count (vectorised)
    featured["total_service_count"] = _compute_service_count(featured)

    # Charge-to-tenure ratio (safe division)
    safe_tenure = featured["tenure"].replace({0: np.nan})
    featured["charge_to_tenure_ratio"] = featured["TotalCharges"] / safe_tenure

    # Average monthly charge (same as ratio but semantically distinct name)
    featured["avg_monthly_charge"] = featured["TotalCharges"] / safe_tenure

    # Contract risk score — month-to-month is highest risk
    contract_risk_map = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1,
    }
    if "Contract" in featured.columns:
        featured["contract_risk_score"] = (
            featured["Contract"]
            .astype(str)
            .map(contract_risk_map)
            .fillna(2)
            .astype(int)
        )

    n_new = sum(
        c in featured.columns
        for c in [
            "tenure_bucket", "total_service_count", "charge_to_tenure_ratio",
            "avg_monthly_charge", "contract_risk_score",
        ]
    )
    logger.info("Added %d engineered features (%d total columns)", n_new, len(featured.columns))
    return featured


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_featured_data(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save featured dataframe to CSV.

    Args:
        df: Featured dataframe.
        output_path: Target file.  Defaults to ``paths.featured_csv``.

    Returns:
        Path where the file was written.
    """
    out = Path(output_path) if output_path is not None else path_config.featured_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved featured data to %s", out)
    return out
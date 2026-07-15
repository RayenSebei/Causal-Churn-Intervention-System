"""Data loading and cleaning helpers for the Telco churn project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PURE_BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]


@dataclass(frozen=True)
class CleanReport:
    """Summary of explicit cleaning actions."""

    rows_before: int
    rows_after: int
    rows_dropped_zero_tenure_totalcharges: int
    rows_dropped_unexpected_missing_totalcharges: int


def resolve_data_path(base_path: str | Path) -> Path:
    """Resolve the dataset path from the workspace root or a provided file path."""

    path = Path(base_path)
    if path.is_dir():
        candidate = path / RAW_DATA_FILENAME
        if candidate.exists():
            return candidate
    return path


def load_telco_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV as strings so missing-value handling stays explicit."""

    path = resolve_data_path(csv_path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    stripped = df.copy()
    for column in stripped.columns:
        stripped[column] = stripped[column].astype(str).str.strip()
    return stripped


def clean_telco_data(df: pd.DataFrame, *, drop_zero_tenure_rows: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    """Clean the Telco churn dataset with explicit handling for TotalCharges."""

    cleaned = _strip_object_columns(df)
    rows_before = len(cleaned)

    numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for column in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"].replace("", pd.NA), errors="coerce")

    zero_tenure_missing_total = cleaned["tenure"].eq(0) & cleaned["TotalCharges"].isna()
    unexpected_missing_total = cleaned["TotalCharges"].isna() & ~zero_tenure_missing_total

    if unexpected_missing_total.any():
        cleaned = cleaned.loc[~unexpected_missing_total].copy()

    dropped_zero_tenure_rows = 0
    if drop_zero_tenure_rows and zero_tenure_missing_total.any():
        dropped_zero_tenure_rows = int(zero_tenure_missing_total.sum())
        cleaned = cleaned.loc[~zero_tenure_missing_total].copy()

    if cleaned["TotalCharges"].isna().any():
        raise ValueError("TotalCharges still contains missing values after explicit cleaning.")

    for column in PURE_BINARY_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace({"Yes": 1, "No": 0})

    service_columns = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for column in service_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace({"No phone service": "No service", "No internet service": "No service"})
            cleaned[column] = cleaned[column].astype("category")

    categorical_columns = [
        "gender",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]
    for column in categorical_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("category")

    cleaned["Churn"] = cleaned["Churn"].astype(int)

    report = CleanReport(
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_dropped_zero_tenure_totalcharges=dropped_zero_tenure_rows,
        rows_dropped_unexpected_missing_totalcharges=int(unexpected_missing_total.sum()),
    )
    return cleaned, report


def save_clean_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def binary_column_summary(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return a compact summary of binary columns after encoding."""

    selected = list(columns) if columns is not None else [column for column in PURE_BINARY_COLUMNS if column in df.columns]
    summary = {
        column: df[column].value_counts(dropna=False).to_dict()
        for column in selected
        if column in df.columns
    }
    return pd.DataFrame(summary).T
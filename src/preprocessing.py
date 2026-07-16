"""Data loading, cleaning, and preprocessing for the Telco churn dataset.

This module replaces the original ``data_prep.py`` with improved type hints,
logging, configurable constants, and vectorized operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import paths as path_config
from src.constants import (
    BINARY_YES_NO_MAP,
    TARGET_COLUMN,
    CUSTOMER_ID_COLUMN,
    NUMERIC_RAW_COLUMNS,
    PURE_BINARY_COLUMNS,
    RAW_DATA_FILENAME,
    SERVICE_COLUMNS,
    SERVICE_NO_MAPPINGS,
    UNORDERED_CATEGORICAL_COLUMNS,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Clean report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CleanReport:
    """Summary of explicit cleaning actions applied to the raw dataset.

    Attributes:
        rows_before: Row count before any cleaning.
        rows_after: Row count after all cleaning steps.
        rows_dropped_zero_tenure_totalcharges: Rows with tenure == 0 and
            blank TotalCharges (expected artefact of the dataset).
        rows_dropped_unexpected_missing_totalcharges: Rows with missing
            TotalCharges that cannot be explained by zero tenure.
    """

    rows_before: int
    rows_after: int
    rows_dropped_zero_tenure_totalcharges: int
    rows_dropped_unexpected_missing_totalcharges: int


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_data_path(base_path: str | Path) -> Path:
    """Resolve the dataset path from a workspace root or a direct file path.

    If *base_path* is a directory, look for the standard raw-data filename
    inside it.  Otherwise treat it as a direct file path.

    Args:
        base_path: Directory or file path.

    Returns:
        Resolved ``Path`` to the CSV file.
    """
    path = Path(base_path)
    if path.is_dir():
        candidate = path / RAW_DATA_FILENAME
        if candidate.exists():
            return candidate
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_telco_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV with all columns as strings.

    Reading as strings keeps missing-value handling fully explicit — no
    silent ``NaN`` injection.

    Args:
        csv_path: Path to the raw CSV (or its parent directory).

    Returns:
        Raw ``DataFrame`` with string dtypes.
    """
    path = resolve_data_path(csv_path)
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every column."""
    stripped = df.copy()
    obj_cols = stripped.select_dtypes(include="object").columns
    stripped[obj_cols] = stripped[obj_cols].apply(lambda col: col.str.strip())
    return stripped


def clean_telco_data(
    df: pd.DataFrame,
    *,
    drop_zero_tenure_rows: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    """Clean the Telco churn dataset with explicit TotalCharges handling.

    Steps:
        1. Strip whitespace from all string columns.
        2. Cast numeric columns (``SeniorCitizen``, ``tenure``,
           ``MonthlyCharges``, ``TotalCharges``) to numeric types.
        3. Handle blank ``TotalCharges`` (tenure == 0 artefact).
        4. Encode binary Yes/No columns to 0/1.
        5. Standardise service-column values and cast to ``category``.
        6. Cast unordered categorical columns to ``category``.

    Args:
        df: Raw dataframe (all-string dtypes from ``load_telco_data``).
        drop_zero_tenure_rows: Whether to drop rows where tenure == 0
            *and* TotalCharges is blank (default ``True``).

    Returns:
        Tuple of (cleaned ``DataFrame``, ``CleanReport``).

    Raises:
        ValueError: If TotalCharges still contains nulls after cleaning.
    """
    cleaned = _strip_object_columns(df)
    rows_before = len(cleaned)

    # --- Numeric coercion ---------------------------------------------------
    for column in NUMERIC_RAW_COLUMNS:
        if column == "TotalCharges":
            cleaned[column] = pd.to_numeric(
                cleaned[column].replace("", pd.NA), errors="coerce",
            )
        else:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # --- TotalCharges missingness -------------------------------------------
    zero_tenure_missing = cleaned["tenure"].eq(0) & cleaned["TotalCharges"].isna()
    unexpected_missing = cleaned["TotalCharges"].isna() & ~zero_tenure_missing

    if unexpected_missing.any():
        n_unexpected = int(unexpected_missing.sum())
        logger.warning(
            "Dropping %d rows with unexplained missing TotalCharges",
            n_unexpected,
        )
        cleaned = cleaned.loc[~unexpected_missing].copy()

    dropped_zero_tenure = 0
    if drop_zero_tenure_rows and zero_tenure_missing.any():
        dropped_zero_tenure = int(zero_tenure_missing.sum())
        logger.info(
            "Dropping %d rows with tenure=0 and blank TotalCharges",
            dropped_zero_tenure,
        )
        cleaned = cleaned.loc[~zero_tenure_missing].copy()

    if cleaned["TotalCharges"].isna().any():
        raise ValueError(
            "TotalCharges still contains missing values after explicit cleaning."
        )

    # --- Binary encoding ----------------------------------------------------
    for column in PURE_BINARY_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(BINARY_YES_NO_MAP)

    # --- Service columns ----------------------------------------------------
    for column in SERVICE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .replace(SERVICE_NO_MAPPINGS)
                .astype("category")
            )

    # --- Unordered categoricals ---------------------------------------------
    for column in UNORDERED_CATEGORICAL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("category")

    # --- Ensure target is int -----------------------------------------------
    # Use the canonical target column name from constants for consistency.
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    report = CleanReport(
        rows_before=rows_before,
        rows_after=len(cleaned),
        rows_dropped_zero_tenure_totalcharges=dropped_zero_tenure,
        rows_dropped_unexpected_missing_totalcharges=int(unexpected_missing.sum()),
    )
    logger.info(
        "Cleaning complete: %d → %d rows (dropped %d)",
        report.rows_before,
        report.rows_after,
        report.rows_before - report.rows_after,
    )
    return cleaned, report


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_clean_data(df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    """Save cleaned dataframe to CSV.

    Args:
        df: Cleaned dataframe.
        output_path: Target path.  Defaults to ``paths.clean_csv``.

    Returns:
        Path where the file was written.
    """
    out = Path(output_path) if output_path is not None else path_config.clean_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved cleaned data to %s", out)
    return out


def binary_column_summary(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a compact summary of binary columns after encoding.

    Args:
        df: DataFrame with binary-encoded columns.
        columns: Specific columns to summarise. Defaults to all
            ``PURE_BINARY_COLUMNS`` present in *df*.

    Returns:
        Summary ``DataFrame`` with value-count distributions.
    """
    selected = (
        list(columns)
        if columns is not None
        else [c for c in PURE_BINARY_COLUMNS if c in df.columns]
    )
    summary = {
        col: df[col].value_counts(dropna=False).to_dict()
        for col in selected
        if col in df.columns
    }
    return pd.DataFrame(summary).T

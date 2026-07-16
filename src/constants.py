"""Domain constants for the Causal Churn Intervention System.

Column names, segment labels, and categorical value mappings that are fixed
by the dataset schema — not tuneable parameters (those live in ``config.py``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

TARGET_COLUMN: str = "Churn"
CUSTOMER_ID_COLUMN: str = "customerID"

# ---------------------------------------------------------------------------
# Binary columns (Yes/No → 1/0)
# ---------------------------------------------------------------------------

PURE_BINARY_COLUMNS: list[str] = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# ---------------------------------------------------------------------------
# Service columns (Yes / No / No phone service / No internet service)
# ---------------------------------------------------------------------------

SERVICE_COLUMNS: list[str] = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ALL_SERVICE_COLUMNS: list[str] = [
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

UNORDERED_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
]

NUMERIC_RAW_COLUMNS: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

TENURE_BIN_EDGES: list[int] = [-1, 6, 12, 24, 48, 72]
TENURE_BIN_LABELS: list[str] = ["0-6", "7-12", "13-24", "25-48", "49-72"]

# ---------------------------------------------------------------------------
# Customer segments (canonical 4-way uplift taxonomy)
# ---------------------------------------------------------------------------

SEGMENT_PERSUADABLES: str = "Persuadables"
SEGMENT_SURE_THINGS: str = "Sure Things"
SEGMENT_LOST_CAUSES: str = "Lost Causes"
SEGMENT_SLEEPING_DOGS: str = "Sleeping Dogs"

# Legacy alias kept for backward compatibility with older dashboards / docs.
# New segmentation maps this cohort into Persuadables (positive uplift).
SEGMENT_LOW_RISK_UPSIDE: str = "Low-Risk Upside"

ALL_SEGMENTS: list[str] = [
    SEGMENT_PERSUADABLES,
    SEGMENT_SURE_THINGS,
    SEGMENT_LOST_CAUSES,
    SEGMENT_SLEEPING_DOGS,
]

TARGET_SEGMENTS: list[str] = [
    SEGMENT_PERSUADABLES,
]

LEGACY_SEGMENT_ALIASES: dict[str, str] = {
    SEGMENT_LOW_RISK_UPSIDE: SEGMENT_PERSUADABLES,
}

SEGMENT_COLORS: dict[str, str] = {
    SEGMENT_PERSUADABLES: "#2ecc71",
    SEGMENT_SURE_THINGS: "#3498db",
    SEGMENT_LOST_CAUSES: "#e74c3c",
    SEGMENT_SLEEPING_DOGS: "#f39c12",
    SEGMENT_LOW_RISK_UPSIDE: "#2ecc71",  # alias colour → Persuadables
}

FALLBACK_SEGMENT: str = SEGMENT_SURE_THINGS

SERVICE_NO_MAPPINGS: dict[str, str] = {
    "No phone service": "No service",
    "No internet service": "No service",
}

BINARY_YES_NO_MAP: dict[str, int] = {"Yes": 1, "No": 0}

RAW_DATA_FILENAME: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

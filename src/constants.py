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

# All service-related columns (for feature engineering)
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

# ---------------------------------------------------------------------------
# Categorical columns (for one-hot encoding)
# ---------------------------------------------------------------------------

UNORDERED_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
]

# ---------------------------------------------------------------------------
# Numeric columns
# ---------------------------------------------------------------------------

NUMERIC_RAW_COLUMNS: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# ---------------------------------------------------------------------------
# Tenure bucketing
# ---------------------------------------------------------------------------

TENURE_BIN_EDGES: list[int] = [-1, 6, 12, 24, 48, 72]
TENURE_BIN_LABELS: list[str] = ["0-6", "7-12", "13-24", "25-48", "49-72"]

# ---------------------------------------------------------------------------
# Customer segments (causal/uplift)
# ---------------------------------------------------------------------------

SEGMENT_PERSUADABLES: str = "Persuadables"
SEGMENT_SURE_THINGS: str = "Sure Things"
SEGMENT_LOST_CAUSES: str = "Lost Causes"
SEGMENT_SLEEPING_DOGS: str = "Sleeping Dogs"
SEGMENT_LOW_RISK_UPSIDE: str = "Low-Risk Upside"

ALL_SEGMENTS: list[str] = [
    SEGMENT_PERSUADABLES,
    SEGMENT_SURE_THINGS,
    SEGMENT_LOST_CAUSES,
    SEGMENT_SLEEPING_DOGS,
    SEGMENT_LOW_RISK_UPSIDE,
]

TARGET_SEGMENTS: list[str] = [
    SEGMENT_PERSUADABLES,
    SEGMENT_LOW_RISK_UPSIDE,
]

# ---------------------------------------------------------------------------
# Segment display colours (dashboard)
# ---------------------------------------------------------------------------

SEGMENT_COLORS: dict[str, str] = {
    SEGMENT_PERSUADABLES: "#2ecc71",
    SEGMENT_SURE_THINGS: "#3498db",
    SEGMENT_LOST_CAUSES: "#e74c3c",
    SEGMENT_SLEEPING_DOGS: "#f39c12",
    SEGMENT_LOW_RISK_UPSIDE: "#9b59b6",
}

# ---------------------------------------------------------------------------
# Service-value mappings used during cleaning
# ---------------------------------------------------------------------------

SERVICE_NO_MAPPINGS: dict[str, str] = {
    "No phone service": "No service",
    "No internet service": "No service",
}

BINARY_YES_NO_MAP: dict[str, int] = {"Yes": 1, "No": 0}

# ---------------------------------------------------------------------------
# Raw data file name
# ---------------------------------------------------------------------------

RAW_DATA_FILENAME: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

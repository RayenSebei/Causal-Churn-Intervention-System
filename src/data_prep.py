"""Backward-compatibility shim — all logic now lives in ``preprocessing.py``.

Existing imports such as ``from src.data_prep import clean_telco_data`` continue
to work.  New code should import from ``src.preprocessing`` directly.
"""

from src.preprocessing import (  # noqa: F401
    CleanReport,
    binary_column_summary,
    clean_telco_data,
    load_telco_data,
    resolve_data_path,
    save_clean_data,
)

# Re-export the constant for any callers that referenced it here.
RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PURE_BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]
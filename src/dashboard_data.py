"""Dashboard data pipeline: combine model, explainability, and uplift results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.causal import run_uplift_pipeline
from src.constants import TARGET_COLUMN
from src.explainability import generate_customer_explanations
from src.training import load_featured_frame


def load_dashboard_data(
    csv_path: str | Path,
    baseline_model_path: str | Path,
) -> dict[str, Any]:
    """Load and integrate all model outputs for the dashboard.

    Splits once, then passes the same X_test/y_test/model into uplift and
    explanations so row alignment cannot silently desync.
    """

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=[TARGET_COLUMN])
    y = featured_df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(baseline_model_path)
    uplift_results = run_uplift_pipeline(X_test, y_test, model)

    X_dash = uplift_results["X_test"]
    y_dash = uplift_results["y_test"]
    baseline_probs = uplift_results["baseline_probs"]
    cate = uplift_results["cate"]
    segments = uplift_results["segments"]

    feature_names = list(X_dash.columns)
    explanations = generate_customer_explanations(
        model, X_dash, y_dash, feature_names, num_examples=None
    )
    explanation_map = {exp["customer_index"]: exp["explanation"] for exp in explanations}

    df_dashboard = X_dash.reset_index(drop=True).copy()
    df_dashboard["churn_probability"] = baseline_probs
    df_dashboard["segment"] = segments.values
    df_dashboard["uplift"] = cate
    df_dashboard["expected_churn_if_treated"] = np.clip(baseline_probs - cate, 0, 1)
    df_dashboard["shap_explanation"] = df_dashboard.index.map(
        lambda pos: explanation_map.get(pos, "No explanation available")
    )
    df_dashboard["y_actual"] = y_dash.values

    missing_rate = float(df_dashboard["shap_explanation"].eq("No explanation available").mean())
    assert missing_rate < 0.01, (
        f"SHAP explanations missing for {missing_rate:.1%} of rows (expected ~0)"
    )

    return {
        "df": df_dashboard,
        "baseline_probs": baseline_probs,
        "uplift_results": uplift_results,
        "model": model,
    }

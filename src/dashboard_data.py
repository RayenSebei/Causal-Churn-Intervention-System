"""Dashboard data pipeline: combine model, explainability, and uplift results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.explain import get_preprocessed_feature_names, generate_customer_explanations
from src.model import load_featured_frame
from src.uplift import run_uplift_pipeline


def load_dashboard_data(
    csv_path: str | Path,
    baseline_model_path: str | Path,
) -> dict[str, Any]:
    """Load and integrate all model outputs for the dashboard.

    Args:
        csv_path: Path to raw Telco CSV.
        baseline_model_path: Path to saved baseline model.

    Returns:
        Dictionary with feature dataframe, predictions, explanations, and segments.
    """

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=["Churn"])
    y = featured_df["Churn"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(baseline_model_path)

    baseline_probs = model.predict_proba(X_test)[:, 1]

    uplift_results = run_uplift_pipeline(csv_path, baseline_model_path)

    feature_names = list(X_test.columns)
    explanations = generate_customer_explanations(model, X_test, y_test, feature_names, num_examples=10)

    explanation_map = {exp["customer_index"]: exp["explanation"] for exp in explanations}

    df_dashboard = X_test.copy()
    df_dashboard["churn_probability"] = baseline_probs
    df_dashboard["segment"] = uplift_results["segments"].values
    df_dashboard["uplift"] = uplift_results["cate"]
    df_dashboard["expected_churn_if_treated"] = np.maximum(0, baseline_probs - uplift_results["cate"])
    df_dashboard["shap_explanation"] = df_dashboard.index.map(
        lambda idx: explanation_map.get(idx, "No explanation available")
    )
    df_dashboard["y_actual"] = y_test.values

    return {
        "df": df_dashboard,
        "baseline_probs": baseline_probs,
        "uplift_results": uplift_results,
        "model": model,
    }


def compute_roi_metrics(
    df: pd.DataFrame,
    discount_cost_per_customer: float = 10.0,
) -> dict[str, float]:
    """Compute ROI for targeted vs. blanket retention spending.

    Args:
        df: Dashboard dataframe with churn_probability, segment, and uplift.
        discount_cost_per_customer: Cost to offer retention discount per customer.

    Returns:
        Dictionary with ROI metrics for both strategies.
    """

    persuadables_sleeping_dogs = df[df["segment"].isin(["Persuadables", "Sleeping Dogs"])]
    persuadables_only = df[df["segment"] == "Persuadables"]

    churn_prevented_targeted = (
        persuadables_sleeping_dogs["churn_probability"].sum() - 
        persuadables_sleeping_dogs["expected_churn_if_treated"].sum()
    )
    cost_targeted = len(persuadables_sleeping_dogs) * discount_cost_per_customer

    churn_prevented_blanket = (
        df["churn_probability"].sum() - 
        df["expected_churn_if_treated"].sum()
    )
    cost_blanket = len(df) * discount_cost_per_customer

    roi_targeted = churn_prevented_targeted / cost_targeted if cost_targeted > 0 else 0
    roi_blanket = churn_prevented_blanket / cost_blanket if cost_blanket > 0 else 0
    roi_improvement = (roi_targeted - roi_blanket) / roi_blanket if roi_blanket > 0 else 0

    return {
        "churn_prevented_targeted": float(churn_prevented_targeted),
        "cost_targeted": float(cost_targeted),
        "roi_targeted": float(roi_targeted),
        "customers_targeted": len(persuadables_sleeping_dogs),
        "churn_prevented_blanket": float(churn_prevented_blanket),
        "cost_blanket": float(cost_blanket),
        "roi_blanket": float(roi_blanket),
        "customers_blanket": len(df),
        "roi_improvement": float(roi_improvement),
    }

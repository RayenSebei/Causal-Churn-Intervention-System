"""Dashboard data pipeline: combine model, explainability, and uplift results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.explain import generate_customer_explanations
from src.model import load_featured_frame
from src.uplift import run_uplift_pipeline


def load_dashboard_data(
    csv_path: str | Path,
    baseline_model_path: str | Path,
) -> dict[str, Any]:
    """Load and integrate all model outputs for the dashboard.

    Splits once, then passes the same X_test/y_test/model into uplift and
    explanations so row alignment cannot silently desync.

    Args:
        csv_path: Path to raw Telco CSV.
        baseline_model_path: Path to saved baseline model.

    Returns:
        Dictionary with feature dataframe, predictions, explanations, and segments.
    """

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=["Churn"])
    y = featured_df["Churn"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(baseline_model_path)

    uplift_results = run_uplift_pipeline(X_test, y_test, model)

    # Uplift pipeline returns the held-out uplift-eval subset — use that for the dashboard.
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

    X_test_reset = X_dash.reset_index(drop=True)
    df_dashboard = X_test_reset.copy()
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


def compute_roi_metrics(
    df: pd.DataFrame,
    discount_cost_per_customer: float = 10.0,
) -> dict[str, float]:
    """Compute retention-efficiency metrics for targeted vs. blanket spend.

    Target group: Persuadables + Low-Risk Upside (excludes Sleeping Dogs, where
    treatment is expected to backfire).

    "ROI" here is churn-prevented per discount dollar, not a dollar return multiple.
    Also reports expected retained monthly revenue using MonthlyCharges when present.

    Args:
        df: Dashboard dataframe with churn_probability, segment, and uplift.
        discount_cost_per_customer: Cost to offer retention discount per customer.

    Returns:
        Dictionary with efficiency and revenue metrics for both strategies.
    """

    target_segments = ["Persuadables", "Low-Risk Upside"]
    targeted = df[df["segment"].isin(target_segments)]

    churn_prevented_targeted = (
        targeted["churn_probability"].sum() - targeted["expected_churn_if_treated"].sum()
    )
    cost_targeted = len(targeted) * discount_cost_per_customer

    churn_prevented_blanket = (
        df["churn_probability"].sum() - df["expected_churn_if_treated"].sum()
    )
    cost_blanket = len(df) * discount_cost_per_customer

    roi_targeted = churn_prevented_targeted / cost_targeted if cost_targeted > 0 else 0.0
    roi_blanket = churn_prevented_blanket / cost_blanket if cost_blanket > 0 else 0.0
    roi_improvement = (roi_targeted - roi_blanket) / roi_blanket if roi_blanket > 0 else 0.0

    def _retained_revenue(subset: pd.DataFrame) -> float:
        if "MonthlyCharges" not in subset.columns or len(subset) == 0:
            return 0.0
        prevented = subset["churn_probability"] - subset["expected_churn_if_treated"]
        return float((prevented * subset["MonthlyCharges"]).sum())

    return {
        "churn_prevented_targeted": float(churn_prevented_targeted),
        "cost_targeted": float(cost_targeted),
        "roi_targeted": float(roi_targeted),
        "customers_targeted": len(targeted),
        "retained_revenue_targeted": _retained_revenue(targeted),
        "churn_prevented_blanket": float(churn_prevented_blanket),
        "cost_blanket": float(cost_blanket),
        "roi_blanket": float(roi_blanket),
        "customers_blanket": len(df),
        "retained_revenue_blanket": _retained_revenue(df),
        "roi_improvement": float(roi_improvement),
    }

"""SHAP-based explainability and interpretation for the churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.model import load_featured_frame


def load_trained_model(model_path: str | Path):
    """Load a saved pipeline."""
    return joblib.load(model_path)


def compute_shap_values(
    model: Any,
    X_sample: pd.DataFrame,
    background_sample_size: int = 100,
) -> tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP Explainer and values for the model.

    Uses tree-path-dependent SHAP (default TreeExplainer). A background sample
    is not required for that mode; background_sample_size is retained for API
    compatibility but intentionally unused.

    Args:
        model: Fitted imblearn Pipeline with preprocessor, SMOTE, and XGBoost.
        X_sample: Feature dataframe to explain (typically test set).
        background_sample_size: Unused (kept for callers); path-dependent SHAP.

    Returns:
        Tuple of (SHAP Explainer, SHAP values array).
    """

    del background_sample_size  # path-dependent SHAP does not use a background set
    preprocessed_sample = model.named_steps["preprocessor"].transform(X_sample)
    xgb_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(preprocessed_sample)
    return explainer, shap_vals


def plot_global_feature_importance(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
) -> Path:
    """Create and save a global SHAP importance plot."""

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def explain_customer_churn(
    customer_index: int,
    shap_values: np.ndarray,
    feature_names: list[str],
    X_original: pd.DataFrame,
    top_k: int = 3,
) -> dict[str, Any]:
    """Generate a plain-English explanation for a single customer's churn risk.

    Args:
        customer_index: Positional row index (0-based) into shap_values / X_original.
        shap_values: SHAP values array (n_samples, n_features) in log-odds (margin) space.
        feature_names: List of feature names (after preprocessing).
        X_original: Original feature dataframe with actual values.
        top_k: Number of top contributing features to include.

    Returns:
        Dictionary with customer_index and explanation text.
    """

    customer_shap = shap_values[customer_index]
    top_indices = np.argsort(np.abs(customer_shap))[-top_k:][::-1]

    explanation_parts = []
    for feat_idx in top_indices:
        feature_name = feature_names[feat_idx]
        shap_value = customer_shap[feat_idx]
        direction = "increases" if shap_value > 0 else "decreases"
        magnitude = abs(shap_value)

        explanation_parts.append(
            f"{feature_name} ({direction} churn risk by {magnitude:.3f} log-odds units)"
        )

    explanation = "; ".join(explanation_parts)
    return {
        "customer_index": customer_index,
        "explanation": explanation,
        "top_features": [feature_names[i] for i in top_indices],
        "top_shap_values": customer_shap[top_indices],
    }


def get_preprocessed_feature_names(model: Any, X_sample: pd.DataFrame) -> list[str]:
    """Extract feature names after preprocessing (including one-hot encoded categories)."""

    preprocessor = model.named_steps["preprocessor"]

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            if hasattr(encoder, "get_feature_names_out"):
                feature_names.extend(encoder.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)

    return feature_names


def generate_customer_explanations(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names_raw: list[str],
    num_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Generate explanations keyed by positional row index into X_test.

    Args:
        model: Fitted pipeline.
        X_test: Test features.
        y_test: Test target labels (unused when explaining all rows; kept for API).
        feature_names_raw: Unused (preprocessed names are derived from the model).
        num_examples: Number of top-predicted-churners to explain. None = every row.

    Returns:
        List of explanation dictionaries with positional customer_index.
    """

    del feature_names_raw  # explanations use preprocessed feature names
    explainer, shap_values = compute_shap_values(model, X_test)
    del explainer
    feature_names = get_preprocessed_feature_names(model, X_test)

    churn_probs = model.predict_proba(X_test)[:, 1]
    n_rows = len(X_test)

    if num_examples is None:
        selected_indices = np.arange(n_rows)
    else:
        churners = np.where(y_test.values == 1)[0]
        if len(churners) == 0:
            selected_indices = np.argsort(churn_probs)[-num_examples:]
        elif len(churners) < num_examples:
            selected_indices = churners
        else:
            selected_indices = churners[np.argsort(churn_probs[churners])[-num_examples:]]

    explanations = []
    for idx in selected_indices:
        exp = explain_customer_churn(int(idx), shap_values, feature_names, X_test, top_k=3)
        exp["churn_probability"] = float(churn_probs[idx])
        explanations.append(exp)

    return explanations


def run_cli() -> None:
    """Convenience entry point for running explainability end to end."""

    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "baseline_churn_model.joblib"

    model = load_trained_model(model_path)
    featured_df = load_featured_frame(base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    from sklearn.model_selection import train_test_split
    X = featured_df.drop(columns=["Churn"])
    y = featured_df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    feature_names = list(X_test.columns)
    explanations = generate_customer_explanations(model, X_test, y_test, feature_names, num_examples=3)

    for i, exp in enumerate(explanations, 1):
        print(f"\nCustomer {i} (Churn Prob: {exp['churn_probability']:.2%})")
        print(f"  {exp['explanation']}")


if __name__ == "__main__":
    run_cli()

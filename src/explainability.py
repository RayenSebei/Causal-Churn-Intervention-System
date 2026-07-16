"""Public explainability API for SHAP-based churn explanations."""

from __future__ import annotations

from src.explain import (
    compute_shap_values,
    explain_customer_churn,
    generate_customer_explanations,
    get_preprocessed_feature_names,
    load_trained_model,
    plot_global_feature_importance,
)

__all__ = [
    "compute_shap_values",
    "explain_customer_churn",
    "generate_customer_explanations",
    "get_preprocessed_feature_names",
    "load_trained_model",
    "plot_global_feature_importance",
]

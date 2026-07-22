"""Dashboard data pipeline: combine model, explainability, and uplift results."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.causal import run_uplift_pipeline
from src.config import causal as causal_config
from src.constants import CUSTOMER_ID_COLUMN, TARGET_COLUMN
from src.explainability import generate_customer_explanations
from src.logging_config import get_logger
from src.training import load_featured_frame
from src.validation import (
    clip_probability,
    normalize_segment_labels,
    treated_churn_probability,
    validate_dashboard_frame,
)

logger = get_logger(__name__)


def _build_dashboard_frame(
    csv_path: str,
    baseline_model_path: str,
) -> dict[str, Any]:
    """Heavy pipeline used by the cached loader."""

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=[TARGET_COLUMN])
    y = featured_df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(baseline_model_path)
    uplift_results = run_uplift_pipeline(X_test, y_test, model)

    X_dash = uplift_results["X_test"]
    y_dash = uplift_results["y_test"]
    baseline_probs = clip_probability(uplift_results["baseline_probs"])
    cate = np.clip(
        np.asarray(uplift_results["cate"], dtype=float),
        causal_config.cate_clip_min,
        causal_config.cate_clip_max,
    )
    segments = normalize_segment_labels(pd.Series(uplift_results["segments"]).reset_index(drop=True))

    feature_names = list(X_dash.columns)
    X_dash_reset = X_dash.reset_index(drop=True)
    y_dash_reset = pd.Series(np.asarray(y_dash), index=X_dash_reset.index)

    explanations = generate_customer_explanations(
        model, X_dash_reset, y_dash_reset, feature_names, num_examples=None
    )
    explanation_map = {int(exp["customer_index"]): exp["explanation"] for exp in explanations}

    # Positional alignment: explanations are keyed by row position after reset.
    expected_positions = set(range(len(X_dash_reset)))
    actual_positions = set(explanation_map)
    if expected_positions != actual_positions:
        missing = sorted(expected_positions - actual_positions)
        raise AssertionError(
            f"SHAP explanation index mismatch; missing {len(missing)} positions "
            f"(examples: {missing[:5]})"
        )

    df_dashboard = X_dash_reset.copy()
    df_dashboard["churn_probability"] = baseline_probs
    df_dashboard["segment"] = segments.to_numpy()
    df_dashboard["uplift"] = cate
    df_dashboard["expected_churn_if_treated"] = treated_churn_probability(baseline_probs, cate)
    df_dashboard["shap_explanation"] = [
        explanation_map[pos] for pos in range(len(df_dashboard))
    ]
    df_dashboard["y_actual"] = y_dash_reset.to_numpy()

    if CUSTOMER_ID_COLUMN not in df_dashboard.columns:
        df_dashboard[CUSTOMER_ID_COLUMN] = [f"CUST-{i:04d}" for i in range(len(df_dashboard))]
    else:
        df_dashboard[CUSTOMER_ID_COLUMN] = (
            df_dashboard[CUSTOMER_ID_COLUMN].astype(str).fillna("UNKNOWN")
        )

    return {
        "df": df_dashboard,
        "baseline_probs": baseline_probs,
        "uplift_results": uplift_results,
        "model": model,
    }


@lru_cache(maxsize=4)
def _load_dashboard_data_cached(
    csv_path: str,
    baseline_model_path: str,
    csv_mtime_ns: int,
    model_mtime_ns: int,
) -> dict[str, Any]:
    """Cache predictions / SHAP / segments for repeated dashboard launches.

    The ``csv_mtime_ns`` and ``model_mtime_ns`` parameters are included in
    the cache key so that a retrained model (or updated CSV) automatically
    invalidates stale cached results without requiring a process restart.
    """

    logger.info(
        "Building dashboard data cache for %s (csv_mtime=%s, model_mtime=%s)",
        csv_path, csv_mtime_ns, model_mtime_ns,
    )
    return _build_dashboard_frame(csv_path, baseline_model_path)


def load_dashboard_data(
    csv_path: str | Path,
    baseline_model_path: str | Path,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load and integrate all model outputs for the dashboard.

    Splits once, then passes the same X_test/y_test/model into uplift and
    explanations so row alignment cannot silently desync.

    Results are cached by (absolute path, mtime) so a retrained model or
    updated CSV automatically invalidates the cache.
    """

    csv_key = str(Path(csv_path).resolve())
    model_key = str(Path(baseline_model_path).resolve())

    if use_cache:
        csv_mtime_ns = Path(csv_key).stat().st_mtime_ns
        model_mtime_ns = Path(model_key).stat().st_mtime_ns
        payload = _load_dashboard_data_cached(csv_key, model_key, csv_mtime_ns, model_mtime_ns)
    else:
        payload = _build_dashboard_frame(csv_key, model_key)

    # Defensive copy so callers cannot mutate the cached frame.
    df = payload["df"].copy()
    uplift_results = dict(payload["uplift_results"])

    report = validate_dashboard_frame(
        df,
        min_cate=causal_config.cate_clip_min,
        max_cate=causal_config.cate_clip_max,
    )
    if not report["ok"]:
        # Auto-repair probabilities / segments then re-validate once.
        df["churn_probability"] = clip_probability(df["churn_probability"])
        df["uplift"] = np.clip(
            df["uplift"].to_numpy(dtype=float),
            causal_config.cate_clip_min,
            causal_config.cate_clip_max,
        )
        df["expected_churn_if_treated"] = treated_churn_probability(
            df["churn_probability"], df["uplift"]
        )
        df["segment"] = normalize_segment_labels(df["segment"])
        report = validate_dashboard_frame(
            df,
            min_cate=causal_config.cate_clip_min,
            max_cate=causal_config.cate_clip_max,
        )

    return {
        "df": df,
        "baseline_probs": clip_probability(payload["baseline_probs"]),
        "uplift_results": uplift_results,
        "model": payload["model"],
        "validation_report": report,
    }


def clear_dashboard_cache() -> None:
    """Clear the in-process dashboard data cache."""

    _load_dashboard_data_cached.cache_clear()

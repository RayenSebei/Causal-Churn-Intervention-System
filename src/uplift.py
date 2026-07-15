"""Causal/uplift modeling layer for retention targeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.model import load_featured_frame


def assign_synthetic_treatment(
    df: pd.DataFrame,
    treatment_prob_high_charge: float = 0.75,
    treatment_prob_month_to_month: float = 0.70,
    treatment_prob_baseline: float = 0.30,
    random_state: int = 42,
) -> pd.Series:
    """Assign retention discount treatment with higher probability for high-risk segments.

    Args:
        df: Feature dataframe with MonthlyCharges and Contract columns.
        treatment_prob_high_charge: P(treatment | MonthlyCharges > median).
        treatment_prob_month_to_month: P(treatment | Contract == Month-to-month).
        treatment_prob_baseline: P(treatment | baseline).
        random_state: Random seed.

    Returns:
        Binary treatment indicator series (0=control, 1=treatment/discount offered).
    """

    np.random.seed(random_state)
    n = len(df)
    treatment_prob = np.ones(n) * treatment_prob_baseline

    high_charge_mask = df["MonthlyCharges"] > df["MonthlyCharges"].median()
    treatment_prob[high_charge_mask] = treatment_prob_high_charge

    month_to_month_mask = df["Contract"] == "Month-to-month"
    treatment_prob[month_to_month_mask] = np.maximum(
        treatment_prob[month_to_month_mask],
        treatment_prob_month_to_month,
    )

    treatment = np.random.binomial(1, treatment_prob)
    return pd.Series(treatment, index=df.index)


def inject_treatment_effect(
    churn_baseline: np.ndarray,
    treatment: np.ndarray,
    cate: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate outcomes with injected heterogeneous treatment effects.

    Args:
        churn_baseline: Baseline churn probability (control arm).
        treatment: Binary treatment indicator.
        cate: Heterogeneous treatment effect (reduction in churn risk if treated).
        random_state: Random seed.

    Returns:
        Tuple of (outcomes, true_cate).
    """

    np.random.seed(random_state)
    churn_factual = churn_baseline.copy()

    for i in range(len(churn_baseline)):
        if treatment[i] == 1:
            churn_factual[i] = np.maximum(0, churn_baseline[i] - cate[i])

    outcomes = np.random.binomial(1, churn_factual)
    return outcomes, cate


def estimate_cate_from_baseline(
    model: Any,
    X_features: pd.DataFrame,
    baseline_probs: np.ndarray,
) -> np.ndarray:
    """Estimate heterogeneous treatment effect magnitude from baseline churn risk.

    High baseline churn → larger potential uplift from treatment.
    Customers with low/stable baseline → low uplift.
    """

    cate = baseline_probs * 0.3 + (1 - baseline_probs) * 0.05
    return np.clip(cate, 0, 1)


def fit_t_learner_cate(
    X: pd.DataFrame,
    T: pd.Series,
    Y: np.ndarray,
    random_state: int = 42,
) -> tuple[tuple[RandomForestRegressor, RandomForestRegressor], np.ndarray]:
    """Fit a T-learner (two separate models: one per arm) to estimate CATE.

    Args:
        X: Feature matrix.
        T: Binary treatment indicator.
        Y: Outcomes (churn indicator).
        random_state: Random seed.

    Returns:
        Tuple of ((control_model, treatment_model), CATE predictions).
    """

    X_numeric = X.select_dtypes(include=[np.number])

    T_array = T.values
    control_mask = T_array == 0
    treatment_mask = T_array == 1

    if control_mask.sum() > 10 and treatment_mask.sum() > 10:
        model_0 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)
        model_1 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)

        model_0.fit(X_numeric[control_mask], Y[control_mask])
        model_1.fit(X_numeric[treatment_mask], Y[treatment_mask])

        pred_0 = model_0.predict(X_numeric)
        pred_1 = model_1.predict(X_numeric)
        cate = pred_0 - pred_1
    else:
        cate = np.zeros(len(X))
        model_0 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)
        model_1 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)

    return (model_0, model_1), cate


def segment_customers_by_cate(
    X: pd.DataFrame,
    baseline_churn: np.ndarray,
    cate: np.ndarray,
    percentiles: tuple[float, float, float] = (33, 67, 90),
) -> pd.Series:
    """Segment customers into four groups based on baseline churn and CATE.

    Args:
        X: Feature matrix.
        baseline_churn: Baseline churn probability (control arm).
        cate: Heterogeneous treatment effect.
        percentiles: (p_low, p_mid, p_high) for segmentation thresholds.

    Returns:
        Segment assignment (string labels).
    """

    p_low, p_mid, p_high = percentiles
    baseline_threshold = np.percentile(baseline_churn, 50)
    cate_low_threshold = np.percentile(cate, p_low)
    cate_high_threshold = np.percentile(cate, p_high)

    segments = np.empty(len(X), dtype=object)

    high_baseline = baseline_churn >= baseline_threshold
    low_cate = cate <= cate_low_threshold
    high_cate = cate >= cate_high_threshold

    segments[(~high_baseline) & (low_cate)] = "Sure Things"
    segments[(high_baseline) & (low_cate)] = "Lost Causes"
    segments[(high_baseline) & (high_cate)] = "Persuadables"
    segments[(~high_baseline) & (~low_cate)] = "Sleeping Dogs"

    return pd.Series(segments, index=X.index)


def run_uplift_pipeline(
    csv_path: str | Path,
    baseline_model_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the end-to-end uplift modeling pipeline.

    Args:
        csv_path: Path to raw Telco CSV.
        baseline_model_path: Path to saved baseline model.
        random_state: Random seed.

    Returns:
        Dictionary with segment counts, examples, and CATE estimates.
    """

    import joblib

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=["Churn"])
    y = featured_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = joblib.load(baseline_model_path)
    baseline_probs_test = model.predict_proba(X_test)[:, 1]
    cate_baseline = estimate_cate_from_baseline(model, X_test, baseline_probs_test)

    treatment_test = assign_synthetic_treatment(X_test, random_state=random_state)
    outcomes_test, true_cate = inject_treatment_effect(baseline_probs_test, treatment_test.values, cate_baseline, random_state=random_state)

    treatment_df = pd.DataFrame({"treatment": treatment_test, "outcome": outcomes_test}, index=X_test.index)

    t_learner_models, cate_learned = fit_t_learner_cate(X_test, treatment_test, outcomes_test, random_state=random_state)

    segments = segment_customers_by_cate(X_test, baseline_probs_test, cate_learned)

    segment_counts = segments.value_counts()
    results = {
        "segment_counts": segment_counts,
        "baseline_probs": baseline_probs_test,
        "cate": cate_learned,
        "segments": segments,
        "X_test": X_test,
        "y_test": y_test,
    }

    segment_examples = {}
    for seg in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]:
        seg_mask = segments == seg
        if seg_mask.sum() > 0:
            seg_indices = np.where(seg_mask)[0]
            top_idx = seg_indices[np.argsort(cate_learned[seg_mask])[-1]]
            segment_examples[seg] = {
                "index": top_idx,
                "baseline_churn": float(baseline_probs_test[top_idx]),
                "uplift": float(cate_learned[top_idx]),
                "expected_churn_if_treated": float(max(0, baseline_probs_test[top_idx] - cate_learned[top_idx])),
            }

    results["segment_examples"] = segment_examples
    return results


def run_cli() -> None:
    """Convenience entry point for uplift modeling."""

    base_dir = Path.cwd().parent
    results = run_uplift_pipeline(
        base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        base_dir / "models" / "baseline_churn_model.joblib",
    )

    print("\n=== Segment Counts ===")
    print(results["segment_counts"])

    print("\n=== Segment Examples (Highest Uplift per Segment) ===")
    for seg, ex in results["segment_examples"].items():
        print(
            f"\n{seg}:"
            f"\n  Baseline Churn: {ex['baseline_churn']:.2%}"
            f"\n  Uplift from Treatment: {ex['uplift']:.2%}"
            f"\n  Expected Churn if Treated: {ex['expected_churn_if_treated']:.2%}"
        )


if __name__ == "__main__":
    run_cli()

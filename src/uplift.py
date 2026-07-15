"""Causal/uplift modeling layer for retention targeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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

    rng = np.random.default_rng(random_state)
    n = len(df)
    treatment_prob = np.ones(n) * treatment_prob_baseline

    high_charge_mask = df["MonthlyCharges"] > df["MonthlyCharges"].median()
    treatment_prob[high_charge_mask] = treatment_prob_high_charge

    month_to_month_mask = df["Contract"] == "Month-to-month"
    treatment_prob[month_to_month_mask] = np.maximum(
        treatment_prob[month_to_month_mask],
        treatment_prob_month_to_month,
    )

    treatment = rng.binomial(1, treatment_prob)
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
        cate: Heterogeneous treatment effect (reduction in churn risk if treated;
            negative values mean treatment increases churn).
        random_state: Random seed.

    Returns:
        Tuple of (outcomes, true_cate).
    """

    rng = np.random.default_rng(random_state)
    churn_factual = churn_baseline.copy()

    for i in range(len(churn_baseline)):
        if treatment[i] == 1:
            churn_factual[i] = float(np.clip(churn_baseline[i] - cate[i], 0, 1))

    outcomes = rng.binomial(1, churn_factual)
    return outcomes, cate


def estimate_cate_from_baseline(
    model: Any,
    X_features: pd.DataFrame,
    baseline_probs: np.ndarray,
) -> np.ndarray:
    """Estimate heterogeneous treatment effect magnitude from baseline churn risk.

    High baseline churn → larger potential uplift from treatment.
    Long-tenure, low-risk customers can have a small negative CATE (backfire).
    """

    cate = baseline_probs * 0.3 + (1 - baseline_probs) * 0.05

    if "tenure" in X_features.columns:
        long_tenure_safe = (X_features["tenure"].to_numpy() > 48) & (baseline_probs < 0.15)
        cate = np.where(long_tenure_safe, -0.03, cate)

    return np.clip(cate, -0.10, 1.0)


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
        Tuple of ((control_model, treatment_model), CATE predictions on X).
    """

    X_numeric = X.select_dtypes(include=[np.number])

    T_array = T.values if hasattr(T, "values") else np.asarray(T)
    control_mask = T_array == 0
    treatment_mask = T_array == 1

    model_0 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)
    model_1 = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state)

    if control_mask.sum() > 10 and treatment_mask.sum() > 10:
        model_0.fit(X_numeric[control_mask], Y[control_mask])
        model_1.fit(X_numeric[treatment_mask], Y[treatment_mask])
        cate = predict_t_learner_cate((model_0, model_1), X)
    else:
        cate = np.zeros(len(X))

    return (model_0, model_1), cate


def predict_t_learner_cate(
    models: tuple[RandomForestRegressor, RandomForestRegressor],
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict CATE with a fitted T-learner (mu0 - mu1)."""

    model_0, model_1 = models
    X_numeric = X.select_dtypes(include=[np.number])
    return model_0.predict(X_numeric) - model_1.predict(X_numeric)


def segment_customers_by_cate(
    X: pd.DataFrame,
    baseline_churn: np.ndarray,
    cate: np.ndarray,
    baseline_percentile: float = 50,
    cate_percentile: float = 50,
) -> pd.Series:
    """Segment customers into exhaustive groups based on baseline churn and CATE.

    Sleeping Dogs = negative learned CATE (treatment backfires — do not target).
    Remaining customers form a 2x2 on high_baseline x high_cate.

    Args:
        X: Feature matrix.
        baseline_churn: Baseline churn probability (control arm).
        cate: Heterogeneous treatment effect.
        baseline_percentile: Percentile cut for high vs low baseline churn.
        cate_percentile: Percentile cut for high vs low CATE (among non-negative).

    Returns:
        Segment assignment (string labels). Guaranteed non-null for every row.
    """

    baseline_threshold = np.percentile(baseline_churn, baseline_percentile)
    cate_threshold = np.percentile(cate, cate_percentile)

    high_baseline = baseline_churn >= baseline_threshold
    high_cate = cate >= cate_threshold
    negative_cate = cate < 0

    segments = np.empty(len(X), dtype=object)
    segments[negative_cate] = "Sleeping Dogs"
    segments[(~negative_cate) & (~high_baseline) & (~high_cate)] = "Sure Things"
    segments[(~negative_cate) & (high_baseline) & (~high_cate)] = "Lost Causes"
    segments[(~negative_cate) & (high_baseline) & (high_cate)] = "Persuadables"
    segments[(~negative_cate) & (~high_baseline) & (high_cate)] = "Low-Risk Upside"

    assert not pd.isna(segments).any(), "Every customer must receive a segment label"
    return pd.Series(segments, index=X.index)


def run_uplift_pipeline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Any,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the uplift modeling pipeline on an already-split test set.

    Further splits X_test into uplift-train / uplift-eval (70/30) so the T-learner
    is not scored in-sample. Trade-off: with only ~1,400 test rows, this shrinks
    segment sizes and dashboard coverage vs. fitting on the full test set.

    Args:
        X_test: Held-out feature matrix from the baseline model split.
        y_test: Held-out labels aligned with X_test.
        model: Fitted baseline churn pipeline.
        random_state: Random seed.

    Returns:
        Dictionary with segment counts, CATE estimates, and recovery metrics
        for the uplift-eval subset.
    """

    baseline_probs_test = model.predict_proba(X_test)[:, 1]
    cate_baseline = estimate_cate_from_baseline(model, X_test, baseline_probs_test)

    treatment_test = assign_synthetic_treatment(X_test, random_state=random_state)
    outcomes_test, true_cate = inject_treatment_effect(
        baseline_probs_test, treatment_test.values, cate_baseline, random_state=random_state
    )

    # Held-out eval for the causal model (avoids in-sample CATE optimism).
    (
        X_up_train,
        X_up_eval,
        treat_train,
        treat_eval,
        out_train,
        out_eval,
        base_train,
        base_eval,
        true_cate_train,
        true_cate_eval,
        y_up_train,
        y_up_eval,
    ) = train_test_split(
        X_test,
        treatment_test,
        outcomes_test,
        baseline_probs_test,
        true_cate,
        y_test,
        test_size=0.3,
        random_state=random_state,
    )

    t_learner_models, _ = fit_t_learner_cate(
        X_up_train, treat_train, out_train, random_state=random_state
    )
    cate_learned = predict_t_learner_cate(t_learner_models, X_up_eval)

    segments = segment_customers_by_cate(X_up_eval, base_eval, cate_learned)

    cate_recovery_corr, _ = pearsonr(true_cate_eval, cate_learned)
    cate_recovery_mae = float(np.mean(np.abs(true_cate_eval - cate_learned)))

    segment_counts = segments.value_counts()
    results: dict[str, Any] = {
        "segment_counts": segment_counts,
        "baseline_probs": base_eval,
        "cate": cate_learned,
        "true_cate": true_cate_eval,
        "segments": segments,
        "X_test": X_up_eval,
        "y_test": y_up_eval,
        "cate_recovery_corr": float(cate_recovery_corr),
        "cate_recovery_mae": cate_recovery_mae,
        "t_learner_models": t_learner_models,
    }

    segment_examples = {}
    for seg in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs", "Low-Risk Upside"]:
        seg_mask = segments == seg
        if seg_mask.sum() > 0:
            seg_indices = np.where(seg_mask.values)[0]
            top_idx = seg_indices[np.argsort(cate_learned[seg_mask.values])[-1]]
            segment_examples[seg] = {
                "index": int(top_idx),
                "baseline_churn": float(base_eval[top_idx]),
                "uplift": float(cate_learned[top_idx]),
                "expected_churn_if_treated": float(
                    np.clip(base_eval[top_idx] - cate_learned[top_idx], 0, 1)
                ),
            }

    results["segment_examples"] = segment_examples
    return results


def run_cli() -> None:
    """Convenience entry point for uplift modeling."""

    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_path = base_dir / "models" / "baseline_churn_model.joblib"

    featured_df = load_featured_frame(csv_path)
    X = featured_df.drop(columns=["Churn"])
    y = featured_df["Churn"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = joblib.load(model_path)

    results = run_uplift_pipeline(X_test, y_test, model)

    print("\n=== Segment Counts ===")
    print(results["segment_counts"])

    print("\n=== CATE Recovery (vs synthetic ground truth) ===")
    print(f"Correlation: {results['cate_recovery_corr']:.3f}")
    print(f"MAE: {results['cate_recovery_mae']:.4f}")

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

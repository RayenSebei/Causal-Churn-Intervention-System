"""Synthetic causal simulation utilities for retention analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from src.causal.learners import LearnerConfig, fit_t_learner_cate, predict_t_learner_cate
from src.causal.policy import segment_customers_by_cate
from src.causal.treatment import TreatmentPolicy, assign_synthetic_treatment
from src.config import causal as causal_config
from src.constants import ALL_SEGMENTS


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the synthetic outcome generation process."""

    random_state: int = 42
    high_risk_multiplier: float = 0.30
    low_risk_floor: float = 0.05
    backfire_effect: float = -0.03
    backfire_tenure_threshold: int = 48
    backfire_churn_threshold: float = 0.15
    cate_clip_min: float = -0.10
    cate_clip_max: float = 1.00


def _default_simulation_config(random_state: int | None = None) -> SimulationConfig:
    """Build a SimulationConfig from central causal settings."""

    return SimulationConfig(
        random_state=causal_config.random_state if random_state is None else random_state,
        high_risk_multiplier=causal_config.high_risk_effect_multiplier,
        low_risk_floor=causal_config.low_risk_effect_floor,
        backfire_effect=causal_config.backfire_effect,
        backfire_tenure_threshold=causal_config.backfire_tenure_threshold,
        backfire_churn_threshold=causal_config.backfire_churn_threshold,
        cate_clip_min=causal_config.cate_clip_min,
        cate_clip_max=causal_config.cate_clip_max,
    )


def inject_treatment_effect(
    churn_baseline: np.ndarray,
    treatment: np.ndarray,
    cate: np.ndarray,
    config: SimulationConfig | None = None,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate observed outcomes from baseline risk and synthetic treatment effect."""

    config = config or _default_simulation_config(random_state)
    rng = np.random.default_rng(config.random_state)

    churn_baseline = np.asarray(churn_baseline, dtype=float)
    treatment = np.asarray(treatment)
    cate = np.asarray(cate, dtype=float)

    treated = treatment == 1
    churn_factual = np.where(treated, np.clip(churn_baseline - cate, 0.0, 1.0), churn_baseline)
    outcomes = rng.binomial(1, churn_factual)
    return outcomes, cate


def estimate_cate_from_baseline(
    model: Any,
    X_features: pd.DataFrame,
    baseline_probs: np.ndarray,
    config: SimulationConfig | None = None,
) -> np.ndarray:
    """Estimate heterogeneous treatment effect magnitude from baseline churn risk.

    High baseline churn → larger potential uplift from treatment.
    Long-tenure, low-risk customers can have a small negative CATE (backfire).
    """

    del model  # reserved for future model-aware effect schedules
    config = config or _default_simulation_config()
    baseline_probs = np.asarray(baseline_probs, dtype=float)
    cate = (
        baseline_probs * config.high_risk_multiplier
        + (1.0 - baseline_probs) * config.low_risk_floor
    )

    if "tenure" in X_features.columns:
        long_tenure_safe = (
            (X_features["tenure"].to_numpy() > config.backfire_tenure_threshold)
            & (baseline_probs < config.backfire_churn_threshold)
        )
        cate = np.where(long_tenure_safe, config.backfire_effect, cate)

    return np.clip(cate, config.cate_clip_min, config.cate_clip_max)


def run_uplift_pipeline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Any,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run the uplift modeling pipeline on an already-split test set.

    Further splits X_test into uplift-train / uplift-eval so the T-learner
    is not scored in-sample. Trade-off: with only ~1,400 test rows, this shrinks
    segment sizes and dashboard coverage vs. fitting on the full test set.
    """

    seed = causal_config.random_state if random_state is None else random_state
    sim_config = _default_simulation_config(seed)

    baseline_probs_test = model.predict_proba(X_test)[:, 1]
    cate_baseline = estimate_cate_from_baseline(model, X_test, baseline_probs_test, sim_config)

    treatment_policy = TreatmentPolicy(
        high_charge_probability=causal_config.treatment_prob_high_charge,
        month_to_month_probability=causal_config.treatment_prob_month_to_month,
        baseline_probability=causal_config.treatment_prob_baseline,
        random_state=seed,
    )
    treatment_test = assign_synthetic_treatment(X_test, policy=treatment_policy)
    outcomes_test, true_cate = inject_treatment_effect(
        baseline_probs_test,
        treatment_test.values,
        cate_baseline,
        config=sim_config,
    )

    (
        X_up_train,
        X_up_eval,
        treat_train,
        _treat_eval,
        out_train,
        _out_eval,
        _base_train,
        base_eval,
        _true_cate_train,
        true_cate_eval,
        _y_up_train,
        y_up_eval,
    ) = train_test_split(
        X_test,
        treatment_test,
        outcomes_test,
        baseline_probs_test,
        true_cate,
        y_test,
        test_size=causal_config.uplift_eval_size,
        random_state=seed,
    )

    learner_config = LearnerConfig(
        n_estimators=causal_config.learner_n_estimators,
        max_depth=causal_config.learner_max_depth,
        random_state=seed,
    )
    t_learner_models, _ = fit_t_learner_cate(
        X_up_train, treat_train, out_train, config=learner_config
    )
    cate_learned = predict_t_learner_cate(t_learner_models, X_up_eval)
    segments = segment_customers_by_cate(
        X_up_eval,
        base_eval,
        cate_learned,
        baseline_percentile=causal_config.baseline_percentile,
        cate_percentile=causal_config.cate_percentile,
    )

    cate_recovery_corr, _ = pearsonr(true_cate_eval, cate_learned)
    cate_recovery_mae = float(np.mean(np.abs(true_cate_eval - cate_learned)))

    results: dict[str, Any] = {
        "segment_counts": segments.value_counts(),
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

    segment_examples: dict[str, dict[str, float | int]] = {}
    for seg in ALL_SEGMENTS:
        seg_mask = segments == seg
        if seg_mask.sum() > 0:
            seg_indices = np.where(seg_mask.to_numpy())[0]
            top_idx = int(seg_indices[np.argsort(cate_learned[seg_mask.to_numpy()])[-1]])
            segment_examples[seg] = {
                "index": top_idx,
                "baseline_churn": float(base_eval[top_idx]),
                "uplift": float(cate_learned[top_idx]),
                "expected_churn_if_treated": float(
                    np.clip(base_eval[top_idx] - cate_learned[top_idx], 0, 1)
                ),
            }

    results["segment_examples"] = segment_examples
    return results

"""Causal learners for uplift estimation.

The learners are intentionally simple sklearn-based implementations so they can
be swapped for EconML / DoWhy later without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class LearnerConfig:
    """Shared learner configuration."""

    n_estimators: int = 100
    max_depth: int = 6
    random_state: int = 42


def _feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature matrix for causal learners."""

    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("Causal learners require at least one numeric feature")
    return numeric


def fit_t_learner_cate(
    X: pd.DataFrame,
    treatment: pd.Series | np.ndarray,
    outcomes: np.ndarray,
    config: LearnerConfig | None = None,
) -> tuple[tuple[RandomForestRegressor, RandomForestRegressor], np.ndarray]:
    """Fit a T-learner and return the fitted models plus CATE estimates."""

    config = config or LearnerConfig()
    X_numeric = _feature_matrix(X)
    treatment_array = np.asarray(treatment)
    control_mask = treatment_array == 0
    treatment_mask = treatment_array == 1

    model_control = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    model_treated = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )

    if control_mask.sum() <= 10 or treatment_mask.sum() <= 10:
        cate = np.zeros(len(X_numeric), dtype=float)
        return (model_control, model_treated), cate

    # Use positional iloc indexing to avoid label-alignment bugs after splits
    # while preserving DataFrame feature names for sklearn.
    control_idx = np.flatnonzero(control_mask)
    treatment_idx = np.flatnonzero(treatment_mask)
    model_control.fit(X_numeric.iloc[control_idx], outcomes[control_mask])
    model_treated.fit(X_numeric.iloc[treatment_idx], outcomes[treatment_mask])
    cate = predict_t_learner_cate((model_control, model_treated), X)
    return (model_control, model_treated), cate


def predict_t_learner_cate(
    models: tuple[RandomForestRegressor, RandomForestRegressor],
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict CATE with a fitted T-learner, clipped to configured bounds."""

    from src.config import causal as causal_config

    model_control, model_treated = models
    X_numeric = _feature_matrix(X)
    raw = model_control.predict(X_numeric) - model_treated.predict(X_numeric)
    return np.clip(raw, causal_config.cate_clip_min, causal_config.cate_clip_max)


def fit_s_learner_cate(
    X: pd.DataFrame,
    treatment: pd.Series | np.ndarray,
    outcomes: np.ndarray,
    base_model: Any | None = None,
    random_state: int = 42,
) -> tuple[Any, np.ndarray]:
    """Fit a simple S-learner with treatment as an additional feature."""

    X_numeric = _feature_matrix(X).copy()
    treatment_array = np.asarray(treatment, dtype=float)
    X_numeric["treatment"] = treatment_array
    model = clone(base_model) if base_model is not None else RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=random_state,
    )
    model.fit(X_numeric, outcomes)

    X_treated = X_numeric.copy()
    X_treated["treatment"] = 1.0
    X_control = X_numeric.copy()
    X_control["treatment"] = 0.0
    cate = model.predict(X_treated) - model.predict(X_control)
    from src.config import causal as causal_config

    return model, np.clip(cate, causal_config.cate_clip_min, causal_config.cate_clip_max)


def predict_s_learner_cate(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict CATE using an S-learner with treatment toggled on/off."""

    from src.config import causal as causal_config

    X_numeric = _feature_matrix(X).copy()
    X_treated = X_numeric.copy()
    X_treated["treatment"] = 1.0
    X_control = X_numeric.copy()
    X_control["treatment"] = 0.0
    raw = model.predict(X_treated) - model.predict(X_control)
    return np.clip(raw, causal_config.cate_clip_min, causal_config.cate_clip_max)


def fit_x_learner_cate(
    X: pd.DataFrame,
    treatment: pd.Series | np.ndarray,
    outcomes: np.ndarray,
    config: LearnerConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit a lightweight X-learner using two outcome models and one effect model."""

    config = config or LearnerConfig()
    X_numeric = _feature_matrix(X)
    treatment_array = np.asarray(treatment)
    control_mask = treatment_array == 0
    treated_mask = treatment_array == 1

    mu0 = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    mu1 = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    control_idx = np.flatnonzero(control_mask)
    treated_idx = np.flatnonzero(treated_mask)
    mu0.fit(X_numeric.iloc[control_idx], outcomes[control_mask])
    mu1.fit(X_numeric.iloc[treated_idx], outcomes[treated_mask])

    tau_control = outcomes[control_mask] - mu1.predict(X_numeric.iloc[control_idx])
    tau_treated = mu0.predict(X_numeric.iloc[treated_idx]) - outcomes[treated_mask]

    tau_model_control = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    tau_model_treated = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )
    tau_model_control.fit(X_numeric.iloc[control_idx], tau_control)
    tau_model_treated.fit(X_numeric.iloc[treated_idx], tau_treated)

    cate = predict_x_learner_cate(
        {"tau_control": tau_model_control, "tau_treated": tau_model_treated},
        X,
    )
    models = {
        "mu0": mu0,
        "mu1": mu1,
        "tau_control": tau_model_control,
        "tau_treated": tau_model_treated,
    }
    return models, cate


def predict_x_learner_cate(models: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Predict CATE with an X-learner by averaging arm-specific effect models."""

    from src.config import causal as causal_config

    X_numeric = _feature_matrix(X)
    tau_control = models["tau_control"].predict(X_numeric)
    tau_treated = models["tau_treated"].predict(X_numeric)
    raw = 0.5 * (tau_control + tau_treated)
    return np.clip(raw, causal_config.cate_clip_min, causal_config.cate_clip_max)

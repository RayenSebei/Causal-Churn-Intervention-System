"""Synthetic causal simulation utilities for retention analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the synthetic outcome generation process."""

    random_state: int = 42
    high_risk_multiplier: float = 0.30
    low_risk_floor: float = 0.05
    backfire_effect: float = -0.03
    backfire_tenure_threshold: int = 48
    backfire_churn_threshold: float = 0.15


def inject_treatment_effect(
    churn_baseline: np.ndarray,
    treatment: np.ndarray,
    cate: np.ndarray,
    config: SimulationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate observed outcomes from baseline risk and synthetic treatment effect.

    The implementation is vectorized and equivalent to the previous loop-based
    version under the same random seed because it produces the same factual
    churn probabilities before the binomial draw.
    """

    config = config or SimulationConfig()
    rng = np.random.default_rng(config.random_state)

    churn_baseline = np.asarray(churn_baseline, dtype=float)
    treatment = np.asarray(treatment)
    cate = np.asarray(cate, dtype=float)

    treated = treatment == 1
    churn_factual = np.where(treated, np.clip(churn_baseline - cate, 0.0, 1.0), churn_baseline)
    outcomes = rng.binomial(1, churn_factual)
    return outcomes, cate

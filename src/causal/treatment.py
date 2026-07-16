"""Treatment assignment primitives for the causal simulation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import causal as causal_config


@dataclass(frozen=True)
class TreatmentPolicy:
    """Configuration for synthetic treatment assignment."""

    high_charge_probability: float = 0.75
    month_to_month_probability: float = 0.70
    baseline_probability: float = 0.30
    random_state: int = 42


def _default_treatment_policy(random_state: int | None = None) -> TreatmentPolicy:
    return TreatmentPolicy(
        high_charge_probability=causal_config.treatment_prob_high_charge,
        month_to_month_probability=causal_config.treatment_prob_month_to_month,
        baseline_probability=causal_config.treatment_prob_baseline,
        random_state=causal_config.random_state if random_state is None else random_state,
    )


def assign_synthetic_treatment(
    df: pd.DataFrame,
    policy: TreatmentPolicy | None = None,
    *,
    treatment_prob_high_charge: float | None = None,
    treatment_prob_month_to_month: float | None = None,
    treatment_prob_baseline: float | None = None,
    random_state: int | None = None,
) -> pd.Series:
    """Assign a binary retention treatment with risk-based probabilities.

    Accepts either a ``TreatmentPolicy`` or the legacy keyword arguments used by
    older callers.
    """

    if policy is None:
        policy = _default_treatment_policy(random_state)
        if any(
            value is not None
            for value in (
                treatment_prob_high_charge,
                treatment_prob_month_to_month,
                treatment_prob_baseline,
                random_state,
            )
        ):
            policy = TreatmentPolicy(
                high_charge_probability=(
                    policy.high_charge_probability
                    if treatment_prob_high_charge is None
                    else treatment_prob_high_charge
                ),
                month_to_month_probability=(
                    policy.month_to_month_probability
                    if treatment_prob_month_to_month is None
                    else treatment_prob_month_to_month
                ),
                baseline_probability=(
                    policy.baseline_probability
                    if treatment_prob_baseline is None
                    else treatment_prob_baseline
                ),
                random_state=policy.random_state if random_state is None else random_state,
            )

    rng = np.random.default_rng(policy.random_state)
    treatment_prob = np.full(len(df), policy.baseline_probability, dtype=float)

    if "MonthlyCharges" in df.columns:
        high_charge_mask = df["MonthlyCharges"] > df["MonthlyCharges"].median()
        treatment_prob[high_charge_mask.to_numpy()] = policy.high_charge_probability

    if "Contract" in df.columns:
        month_to_month_mask = df["Contract"].astype(str).eq("Month-to-month")
        treatment_prob[month_to_month_mask.to_numpy()] = np.maximum(
            treatment_prob[month_to_month_mask.to_numpy()],
            policy.month_to_month_probability,
        )

    treatment = rng.binomial(1, treatment_prob)
    return pd.Series(treatment, index=df.index, name="treatment")

"""Treatment assignment primitives for the causal simulation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreatmentPolicy:
    """Configuration for synthetic treatment assignment."""

    high_charge_probability: float = 0.75
    month_to_month_probability: float = 0.70
    baseline_probability: float = 0.30
    random_state: int = 42


def assign_synthetic_treatment(
    df: pd.DataFrame,
    policy: TreatmentPolicy | None = None,
) -> pd.Series:
    """Assign a binary retention treatment with risk-based probabilities."""

    policy = policy or TreatmentPolicy()
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

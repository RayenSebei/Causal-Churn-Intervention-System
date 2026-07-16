"""ROI and intervention economics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InterventionSpec:
    """Description of a single retention intervention."""

    name: str
    cost_per_customer: float
    acceptance_probability: float
    retention_multiplier: float = 1.0
    description: str = ""


def expected_retained_revenue(
    monthly_charges: pd.Series | np.ndarray,
    baseline_churn: pd.Series | np.ndarray,
    treatment_effect: pd.Series | np.ndarray,
    acceptance_probability: float,
    annual_revenue_multiplier: int = 12,
) -> float:
    """Expected retained revenue from a set of customers."""

    monthly_charges = np.asarray(monthly_charges, dtype=float)
    baseline_churn = np.asarray(baseline_churn, dtype=float)
    treatment_effect = np.asarray(treatment_effect, dtype=float)
    retained_probability = np.clip(baseline_churn - treatment_effect, 0.0, 1.0)
    improvement = np.clip(baseline_churn - retained_probability, 0.0, 1.0)
    return float((monthly_charges * annual_revenue_multiplier * improvement * acceptance_probability).sum())


def expected_campaign_cost(
    cost_per_customer: float,
    customers_targeted: int,
    acceptance_probability: float,
) -> float:
    """Expected campaign spend for a targeting policy."""

    return float(cost_per_customer * customers_targeted * acceptance_probability)


def expected_profit(
    retained_revenue: float,
    campaign_cost: float,
) -> float:
    """Expected net profit of an intervention campaign."""

    return float(retained_revenue - campaign_cost)


def compute_intervention_roi(
    retained_revenue: float,
    campaign_cost: float,
) -> float:
    """Return ROI percentage for an intervention campaign."""

    if campaign_cost <= 0:
        return 0.0
    return float((retained_revenue - campaign_cost) / campaign_cost)

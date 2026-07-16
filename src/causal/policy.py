"""Intervention policy optimization utilities."""

from __future__ import annotations

import pandas as pd

from src.causal.roi import InterventionSpec, compute_intervention_roi, expected_campaign_cost, expected_profit


def rank_customers_for_intervention(
    df: pd.DataFrame,
    *,
    benefit_column: str = "uplift",
    cost_column: str = "intervention_cost",
) -> pd.DataFrame:
    """Rank customers by expected net benefit per unit cost."""

    ranked = df.copy()
    ranked["expected_net_benefit"] = ranked[benefit_column] - ranked[cost_column]
    ranked = ranked.sort_values(by=["expected_net_benefit", benefit_column], ascending=[False, False])
    return ranked


def select_budget_constrained_interventions(
    df: pd.DataFrame,
    budget: float,
    *,
    benefit_column: str = "uplift",
    cost_column: str = "intervention_cost",
) -> pd.DataFrame:
    """Select a top-ranked subset of customers within a campaign budget."""

    ranked = rank_customers_for_intervention(df, benefit_column=benefit_column, cost_column=cost_column)
    cumulative_cost = ranked[cost_column].cumsum()
    return ranked.loc[cumulative_cost <= budget].copy()

"""Intervention policy optimization and customer segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.roi import InterventionSpec, compute_intervention_roi, expected_campaign_cost, expected_profit
from src.constants import (
    SEGMENT_LOST_CAUSES,
    SEGMENT_LOW_RISK_UPSIDE,
    SEGMENT_PERSUADABLES,
    SEGMENT_SLEEPING_DOGS,
    SEGMENT_SURE_THINGS,
)


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
    """

    baseline_churn = np.asarray(baseline_churn, dtype=float)
    cate = np.asarray(cate, dtype=float)

    baseline_threshold = np.percentile(baseline_churn, baseline_percentile)
    cate_threshold = np.percentile(cate, cate_percentile)

    high_baseline = baseline_churn >= baseline_threshold
    high_cate = cate >= cate_threshold
    negative_cate = cate < 0

    segments = np.empty(len(X), dtype=object)
    segments[negative_cate] = SEGMENT_SLEEPING_DOGS
    segments[(~negative_cate) & (~high_baseline) & (~high_cate)] = SEGMENT_SURE_THINGS
    segments[(~negative_cate) & (high_baseline) & (~high_cate)] = SEGMENT_LOST_CAUSES
    segments[(~negative_cate) & (high_baseline) & (high_cate)] = SEGMENT_PERSUADABLES
    segments[(~negative_cate) & (~high_baseline) & (high_cate)] = SEGMENT_LOW_RISK_UPSIDE

    assert not pd.isna(segments).any(), "Every customer must receive a segment label"
    return pd.Series(segments, index=X.index, name="segment")


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


# Re-export ROI helpers commonly used alongside policy selection.
__all__ = [
    "segment_customers_by_cate",
    "rank_customers_for_intervention",
    "select_budget_constrained_interventions",
    "InterventionSpec",
    "compute_intervention_roi",
    "expected_campaign_cost",
    "expected_profit",
]

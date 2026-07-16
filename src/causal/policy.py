"""Intervention policy optimization and customer segmentation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.causal.roi import InterventionSpec, compute_intervention_roi, expected_campaign_cost, expected_profit
from src.constants import (
    ALL_SEGMENTS,
    FALLBACK_SEGMENT,
    SEGMENT_LOST_CAUSES,
    SEGMENT_PERSUADABLES,
    SEGMENT_SLEEPING_DOGS,
    SEGMENT_SURE_THINGS,
)
from src.logging_config import get_logger
from src.validation import normalize_segment_labels, summarize_segment_distribution

logger = get_logger(__name__)


def segment_customers_by_cate(
    X: pd.DataFrame,
    baseline_churn: np.ndarray,
    cate: np.ndarray,
    baseline_percentile: float = 50,
    cate_percentile: float = 50,
) -> pd.Series:
    """Segment customers into an exhaustive 4-way uplift taxonomy.

    Canonical segments
    ------------------
    - Sleeping Dogs: negative CATE (treatment backfires — do not target)
    - Persuadables: non-negative CATE at/above the CATE threshold (includes the
      former "Low-Risk Upside" cohort for a stable 4-label schema)
    - Lost Causes: high baseline churn, below-threshold CATE
    - Sure Things: low baseline churn, below-threshold CATE

    Every row receives a label. Unclassifiable rows are logged and assigned
    ``FALLBACK_SEGMENT``.
    """

    n = len(X)
    baseline_churn = np.asarray(baseline_churn, dtype=float).reshape(-1)
    cate = np.asarray(cate, dtype=float).reshape(-1)

    if len(baseline_churn) != n or len(cate) != n:
        raise ValueError(
            f"Length mismatch: X={n}, baseline={len(baseline_churn)}, cate={len(cate)}"
        )

    # NaN CATE / baseline cannot be classified — treat as zero effect / median risk
    if np.isnan(cate).any():
        n_nan = int(np.isnan(cate).sum())
        logger.warning("Replacing %s NaN CATE value(s) with 0.0 before segmentation", n_nan)
        cate = np.nan_to_num(cate, nan=0.0)
    if np.isnan(baseline_churn).any():
        fill = float(np.nanmedian(baseline_churn))
        n_nan = int(np.isnan(baseline_churn).sum())
        logger.warning("Replacing %s NaN baseline value(s) with median %.4f", n_nan, fill)
        baseline_churn = np.nan_to_num(baseline_churn, nan=fill)

    baseline_threshold = float(np.percentile(baseline_churn, baseline_percentile))
    cate_threshold = float(np.percentile(cate, cate_percentile))

    high_baseline = baseline_churn >= baseline_threshold
    high_cate = cate >= cate_threshold
    negative_cate = cate < 0

    segments = np.full(n, None, dtype=object)
    segments[negative_cate] = SEGMENT_SLEEPING_DOGS
    # All non-negative high-CATE customers are Persuadables (4-label schema).
    segments[(~negative_cate) & high_cate] = SEGMENT_PERSUADABLES
    segments[(~negative_cate) & (~high_cate) & high_baseline] = SEGMENT_LOST_CAUSES
    segments[(~negative_cate) & (~high_cate) & (~high_baseline)] = SEGMENT_SURE_THINGS

    unclassified = pd.isna(segments) | (segments == None)  # noqa: E711
    if unclassified.any():
        n_bad = int(unclassified.sum())
        logger.warning(
            "Assigning fallback segment '%s' to %s unclassified customer(s)",
            FALLBACK_SEGMENT,
            n_bad,
        )
        warnings.warn(
            f"{n_bad} customers could not be classified; assigned '{FALLBACK_SEGMENT}'",
            RuntimeWarning,
            stacklevel=2,
        )
        segments[unclassified] = FALLBACK_SEGMENT

    series = pd.Series(segments, index=X.index, name="segment")
    series = normalize_segment_labels(series)

    if series.isna().any() or (~series.isin(ALL_SEGMENTS)).any():
        raise AssertionError("Every customer must receive a canonical segment label")

    summary = summarize_segment_distribution(series)
    logger.info("Segment distribution: %s", summary)
    # Attach summary for callers that want it without recomputing.
    series.attrs["distribution"] = summary
    return series


def rank_customers_for_intervention(
    df: pd.DataFrame,
    *,
    benefit_column: str = "uplift",
    cost_column: str = "intervention_cost",
) -> pd.DataFrame:
    """Rank customers by expected net benefit per unit cost."""

    ranked = df.copy()
    ranked["expected_net_benefit"] = ranked[benefit_column] - ranked[cost_column]
    ranked = ranked.sort_values(
        by=["expected_net_benefit", benefit_column], ascending=[False, False]
    )
    return ranked


def select_budget_constrained_interventions(
    df: pd.DataFrame,
    budget: float,
    *,
    benefit_column: str = "uplift",
    cost_column: str = "intervention_cost",
) -> pd.DataFrame:
    """Select a top-ranked subset of customers within a campaign budget."""

    ranked = rank_customers_for_intervention(
        df, benefit_column=benefit_column, cost_column=cost_column
    )
    cumulative_cost = ranked[cost_column].cumsum()
    return ranked.loc[cumulative_cost <= budget].copy()


__all__ = [
    "segment_customers_by_cate",
    "rank_customers_for_intervention",
    "select_budget_constrained_interventions",
    "InterventionSpec",
    "compute_intervention_roi",
    "expected_campaign_cost",
    "expected_profit",
]

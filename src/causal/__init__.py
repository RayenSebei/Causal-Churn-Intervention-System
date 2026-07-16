"""Causal inference utilities for churn targeting."""

from src.causal.learners import (
    fit_s_learner_cate,
    fit_t_learner_cate,
    fit_x_learner_cate,
    predict_s_learner_cate,
    predict_t_learner_cate,
    predict_x_learner_cate,
)
from src.causal.policy import (
    rank_customers_for_intervention,
    select_budget_constrained_interventions,
)
from src.causal.roi import (
    InterventionSpec,
    compute_intervention_roi,
    expected_profit,
)
from src.causal.simulation import (
    assign_synthetic_treatment,
    inject_treatment_effect,
)

__all__ = [
    "assign_synthetic_treatment",
    "inject_treatment_effect",
    "fit_s_learner_cate",
    "fit_t_learner_cate",
    "fit_x_learner_cate",
    "predict_s_learner_cate",
    "predict_t_learner_cate",
    "predict_x_learner_cate",
    "rank_customers_for_intervention",
    "select_budget_constrained_interventions",
    "InterventionSpec",
    "compute_intervention_roi",
    "expected_profit",
]

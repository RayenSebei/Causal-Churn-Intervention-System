"""Causal inference utilities for churn targeting."""

from src.causal.learners import (
    LearnerConfig,
    fit_s_learner_cate,
    fit_t_learner_cate,
    fit_x_learner_cate,
    predict_s_learner_cate,
    predict_t_learner_cate,
    predict_x_learner_cate,
)
from src.causal.policy import (
    rank_customers_for_intervention,
    segment_customers_by_cate,
    select_budget_constrained_interventions,
)
from src.causal.roi import (
    InterventionSpec,
    compute_intervention_roi,
    expected_campaign_cost,
    expected_profit,
    expected_retained_revenue,
)
from src.causal.simulation import (
    SimulationConfig,
    estimate_cate_from_baseline,
    inject_treatment_effect,
    run_uplift_pipeline,
)
from src.causal.treatment import (
    TreatmentPolicy,
    assign_synthetic_treatment,
)

__all__ = [
    "LearnerConfig",
    "SimulationConfig",
    "TreatmentPolicy",
    "InterventionSpec",
    "assign_synthetic_treatment",
    "inject_treatment_effect",
    "estimate_cate_from_baseline",
    "run_uplift_pipeline",
    "fit_s_learner_cate",
    "fit_t_learner_cate",
    "fit_x_learner_cate",
    "predict_s_learner_cate",
    "predict_t_learner_cate",
    "predict_x_learner_cate",
    "segment_customers_by_cate",
    "rank_customers_for_intervention",
    "select_budget_constrained_interventions",
    "compute_intervention_roi",
    "expected_campaign_cost",
    "expected_profit",
    "expected_retained_revenue",
]

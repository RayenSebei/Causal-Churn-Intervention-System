"""Centralized configuration for the Causal Churn Intervention System.

Every tunable constant lives here. Modules import from this file instead
of embedding magic numbers.  Override at runtime by mutating the module-level
dataclass instances (e.g. ``config.model.n_estimators = 500``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """File-system locations for data, models, and outputs."""

    project_root: Path = field(default_factory=_project_root)

    @property
    def raw_csv(self) -> Path:
        return self.project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    @property
    def clean_csv(self) -> Path:
        return self.project_root / "data" / "telco_clean.csv"

    @property
    def featured_csv(self) -> Path:
        return self.project_root / "data" / "telco_features.csv"

    @property
    def model_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def baseline_model(self) -> Path:
        return self.model_dir / "baseline_churn_model.joblib"

    @property
    def eda_dir(self) -> Path:
        return self.project_root / "data" / "eda"

    @property
    def calibration_plot(self) -> Path:
        return self.eda_dir / "calibration_curve.png"


# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """XGBoost and pipeline hyper-parameters."""

    n_estimators: int = 250
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    min_child_weight: int = 1
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    tree_method: str = "hist"
    random_state: int = 42
    n_jobs: int = -1
    test_size: float = 0.20
    cv_folds: int = 5
    smote_random_state: int = 42
    classification_threshold: float = 0.50


# ---------------------------------------------------------------------------
# Causal / uplift configuration
# ---------------------------------------------------------------------------

@dataclass
class CausalConfig:
    """Parameters for causal simulation and learner training."""

    # Treatment assignment probabilities
    treatment_prob_high_charge: float = 0.75
    treatment_prob_month_to_month: float = 0.70
    treatment_prob_baseline: float = 0.30

    # Simulation effect magnitudes (probability-point effects).
    # Rationale: real retention offers rarely move churn by more than ~20–30pp;
    # unbounded T-learner outputs (60–90% "uplift") are estimation artefacts.
    high_risk_effect_multiplier: float = 0.25
    low_risk_effect_floor: float = 0.03
    backfire_effect: float = -0.03
    backfire_tenure_threshold: int = 48
    backfire_churn_threshold: float = 0.15
    # Learned / synthetic CATE bounds (MIN_CATE / MAX_CATE)
    cate_clip_min: float = -0.20
    cate_clip_max: float = 0.30

    # T-learner / X-learner base estimator
    learner_n_estimators: int = 100
    learner_max_depth: int = 6

    # Segmentation percentile thresholds
    baseline_percentile: float = 50.0
    cate_percentile: float = 50.0

    # Uplift train/eval split inside the test set
    uplift_eval_size: float = 0.30

    random_state: int = 42


# ---------------------------------------------------------------------------
# Campaign / ROI configuration
# ---------------------------------------------------------------------------

@dataclass
class InterventionSpec:
    """Describes a single retention intervention type."""

    name: str
    cost_per_customer: float
    acceptance_probability: float
    description: str = ""


@dataclass
class CampaignConfig:
    """Budget and intervention parameters for ROI computation."""

    total_budget: float = 50_000.0
    default_discount_cost: float = 10.0
    annual_revenue_multiplier: int = 12  # monthly → annual

    interventions: List[InterventionSpec] = field(default_factory=lambda: [
        InterventionSpec(
            name="discount",
            cost_per_customer=10.0,
            acceptance_probability=0.60,
            description="10% recurring discount for 3 months",
        ),
        InterventionSpec(
            name="upgrade",
            cost_per_customer=25.0,
            acceptance_probability=0.40,
            description="Free service-tier upgrade for 1 month",
        ),
        InterventionSpec(
            name="support",
            cost_per_customer=15.0,
            acceptance_probability=0.50,
            description="Proactive phone support call",
        ),
        InterventionSpec(
            name="loyalty",
            cost_per_customer=20.0,
            acceptance_probability=0.45,
            description="Loyalty reward (points / gift card)",
        ),
    ])

    def get_intervention(self, name: str) -> InterventionSpec:
        """Look up an intervention by name."""
        for spec in self.interventions:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown intervention: {name!r}")


# ---------------------------------------------------------------------------
# Plot / dashboard configuration
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Visual defaults for matplotlib and Plotly."""

    dpi: int = 160
    figsize_default: tuple[int, int] = (10, 6)
    figsize_small: tuple[int, int] = (7, 6)
    calibration_bins: int = 10
    histogram_bins: int = 30
    max_table_rows: int = 50


# ---------------------------------------------------------------------------
# Top-level convenience singletons
# ---------------------------------------------------------------------------

paths = PathConfig()
model = ModelConfig()
causal = CausalConfig()
campaign = CampaignConfig()
plots = PlotConfig()

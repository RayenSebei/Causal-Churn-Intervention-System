"""Baseline churn prediction model training and evaluation.

This module contains a lightweight training entry-point for the baseline
XGBoost classifier. Evaluation plotting and metric computations have been
consolidated into ``src.evaluation`` to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.config import model as model_config
from src.constants import TARGET_COLUMN, CUSTOMER_ID_COLUMN
from src.data_prep import clean_telco_data, load_telco_data
from src.evaluation import compute_metrics, plot_calibration_curve
from src.features import add_feature_columns
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BaselineMetrics:
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    test_roc_auc: float
    test_average_precision: float
    test_precision: float
    test_recall: float
    test_brier_score: float


def load_featured_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load and prepare the training frame from the raw CSV.

    Args:
        csv_path: Path to the raw Telco CSV (or its parent directory).

    Returns:
        Featured DataFrame ready for modelling.
    """
    raw_df = load_telco_data(csv_path)
    clean_df, _ = clean_telco_data(raw_df)
    featured_df = add_feature_columns(clean_df)
    return featured_df


def build_preprocessor(feature_frame: pd.DataFrame):
    """Create a preprocessing transformer for mixed tabular features.

    This helper inspects dtypes and separates numeric / categorical columns
    while excluding the target and identifier columns.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    excluded = {TARGET_COLUMN, CUSTOMER_ID_COLUMN}
    numeric_columns = [
        column
        for column in feature_frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    categorical_columns = [
        column
        for column in feature_frame.columns
        if column not in excluded and not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def build_classifier() -> XGBClassifier:
    """Return the baseline classifier using centrally-configured hyperparams."""
    return XGBClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        learning_rate=model_config.learning_rate,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        reg_lambda=model_config.reg_lambda,
        min_child_weight=model_config.min_child_weight,
        objective=model_config.objective,
        eval_metric=model_config.eval_metric,
        tree_method=model_config.tree_method,
        random_state=model_config.random_state,
        n_jobs=model_config.n_jobs,
    )


def make_pipeline(preprocessor) -> ImbPipeline:
    """Create a training pipeline that applies preprocessing, SMOTE, and model.

    Note: the pipeline produced here is intended for training only; the
    inference pipeline saved for serving should omit SMOTE.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=model_config.smote_random_state)),
            ("model", build_classifier()),
        ]
    )


def split_features_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    return features, target


def train_baseline_model(
    csv_path: str | Path,
    *,
    model_output_path: str | Path,
    calibration_output_path: str | Path,
    random_state: int = 42,
) -> tuple[BaselineMetrics, dict[str, Any]]:
    """Train the baseline churn model and produce evaluation artefacts.

    This function performs an in-memory CV loop, fits on the full training
    set, and returns a small metrics bundle plus artifacts (saved model,
    calibration plot, classification report text, and test arrays).
    """
    featured_df = load_featured_frame(csv_path)
    X, y = split_features_targets(featured_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_config.test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor, _, _ = build_preprocessor(pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1))
    pipeline = make_pipeline(preprocessor)

    cv = StratifiedKFold(n_splits=model_config.cv_folds, shuffle=True, random_state=random_state)
    cv_scores: list[float] = []
    for train_index, valid_index in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]
        y_fold_valid = y_train.iloc[valid_index]

        pipeline.fit(X_fold_train, y_fold_train)
        valid_prob = pipeline.predict_proba(X_fold_valid)[:, 1]
        from sklearn.metrics import roc_auc_score

        cv_scores.append(roc_auc_score(y_fold_valid, valid_prob))

    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Use the reusable evaluation helpers
    eval_metrics = compute_metrics(y_test.values, test_prob, threshold=model_config.classification_threshold)

    metrics = BaselineMetrics(
        cv_roc_auc_mean=float(np.mean(cv_scores)),
        cv_roc_auc_std=float(np.std(cv_scores, ddof=1)),
        test_roc_auc=float(eval_metrics.roc_auc),
        test_average_precision=float(eval_metrics.pr_auc),
        test_precision=float(eval_metrics.precision),
        test_recall=float(eval_metrics.recall),
        test_brier_score=float(eval_metrics.brier_score),
    )

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the training pipeline (contains SMOTE). For serving, callers may
    # want to reconstruct an inference-only pipeline consisting of the
    # preprocessor + model.
    joblib.dump(pipeline, model_output_path)

    calibration_path = plot_calibration_curve(y_test.values, test_prob, output_path=calibration_output_path)

    from sklearn.metrics import classification_report

    artifacts = {
        "model_path": str(model_output_path),
        "calibration_path": str(calibration_path),
        "classification_report": classification_report(y_test, (test_prob >= model_config.classification_threshold).astype(int), zero_division=0),
        "y_test": y_test,
        "y_prob": test_prob,
    }
    return metrics, artifacts


def run_cli() -> None:
    """Convenience entry point for ad hoc training."""
    from src.config import paths

    base_dir = Path(__file__).parent.parent
    metrics, artifacts = train_baseline_model(
        paths.raw_csv,
        model_output_path=base_dir / "models" / "baseline_churn_model.joblib",
        calibration_output_path=base_dir / "data" / "eda" / "phase2_calibration_curve.png",
    )
    print(metrics)
    print(artifacts["classification_report"])


if __name__ == "__main__":
    run_cli()

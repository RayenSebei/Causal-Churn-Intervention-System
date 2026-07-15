"""Baseline churn prediction model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_prep import clean_telco_data, load_telco_data
from src.features import add_feature_columns


TARGET_COLUMN = "Churn"


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
    """Load and prepare the training frame from the raw CSV."""

    raw_df = load_telco_data(csv_path)
    clean_df, _ = clean_telco_data(raw_df)
    featured_df = add_feature_columns(clean_df)
    return featured_df


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create a preprocessing transformer for mixed tabular features."""

    excluded = {TARGET_COLUMN, "customerID"}
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
    """Return the baseline classifier."""

    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def make_pipeline(preprocessor: ColumnTransformer) -> ImbPipeline:
    """SMOTE is applied only after preprocessing, inside training folds.

    This avoids contaminating validation/test data while addressing class imbalance
    more directly than class weights for the initial baseline.
    """

    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", build_classifier()),
        ]
    )


def split_features_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    return features, target


def evaluate_threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_brier_score": brier_score_loss(y_true, y_prob),
    }


def plot_calibration(y_true: pd.Series, y_prob: np.ndarray, output_path: str | Path) -> Path:
    probability_true, probability_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(probability_pred, probability_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def train_baseline_model(
    csv_path: str | Path,
    *,
    model_output_path: str | Path,
    calibration_output_path: str | Path,
    random_state: int = 42,
) -> tuple[BaselineMetrics, dict[str, Any]]:
    featured_df = load_featured_frame(csv_path)
    X, y = split_features_targets(featured_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    preprocessor, _, _ = build_preprocessor(pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1))
    pipeline = make_pipeline(preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores: list[float] = []
    for train_index, valid_index in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]
        y_fold_valid = y_train.iloc[valid_index]

        pipeline.fit(X_fold_train, y_fold_train)
        valid_prob = pipeline.predict_proba(X_fold_valid)[:, 1]
        cv_scores.append(roc_auc_score(y_fold_valid, valid_prob))

    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_average_precision": average_precision_score(y_test, test_prob),
    }
    test_metrics.update(evaluate_threshold_metrics(y_test, test_prob))

    metrics = BaselineMetrics(
        cv_roc_auc_mean=float(np.mean(cv_scores)),
        cv_roc_auc_std=float(np.std(cv_scores, ddof=1)),
        test_roc_auc=float(test_metrics["test_roc_auc"]),
        test_average_precision=float(test_metrics["test_average_precision"]),
        test_precision=float(test_metrics["test_precision"]),
        test_recall=float(test_metrics["test_recall"]),
        test_brier_score=float(test_metrics["test_brier_score"]),
    )

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)

    calibration_path = plot_calibration(y_test, test_prob, calibration_output_path)

    artifacts = {
        "model_path": str(model_output_path),
        "calibration_path": str(calibration_path),
        "classification_report": classification_report(y_test, (test_prob >= 0.5).astype(int), zero_division=0),
        "y_test": y_test,
        "y_prob": test_prob,
    }
    return metrics, artifacts


def run_cli() -> None:
    """Convenience entry point for ad hoc training."""

    base_dir = Path.cwd().parent
    metrics, artifacts = train_baseline_model(
        base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path=base_dir / "models" / "baseline_churn_model.joblib",
        calibration_output_path=base_dir / "data" / "eda" / "phase2_calibration_curve.png",
    )
    print(metrics)
    print(artifacts["classification_report"])


if __name__ == "__main__":
    run_cli()

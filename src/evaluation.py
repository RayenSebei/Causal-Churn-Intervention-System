"""Reusable model evaluation suite for binary classification.

Provides functions for computing metrics, generating plots, tuning
thresholds, and producing comprehensive evaluation reports.  All
functions are stateless and accept arrays — not coupled to any
particular model or pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import plots as plot_config
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationMetrics:
    """Comprehensive metric set for a binary classifier.

    All probability-based metrics use the raw predicted probability; all
    threshold-based metrics use the given *threshold*.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the Precision–Recall curve.
        precision: Precision at threshold.
        recall: Recall at threshold.
        f1: F1 score at threshold.
        accuracy: Accuracy at threshold.
        brier_score: Brier score (calibration loss).
        threshold: Decision threshold used for hard predictions.
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    brier_score: float
    threshold: float


@dataclass
class EvaluationReport:
    """Full evaluation bundle returned by ``evaluate_model``.

    Contains scalar metrics, cross-validation scores, the
    classification report text, and optional artefact paths for saved
    plots.
    """

    metrics: ClassificationMetrics
    classification_report_text: str
    cv_roc_auc_mean: float | None = None
    cv_roc_auc_std: float | None = None
    optimal_threshold_f1: float | None = None
    optimal_threshold_youden: float | None = None
    plot_paths: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the full metric set for a binary classifier.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision boundary for hard predictions.

    Returns:
        ``ClassificationMetrics`` instance.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find the threshold that maximises F1 score via grid search.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        n_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        Optimal threshold (float in [0, 1]).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    logger.info("Optimal F1 threshold: %.3f (F1 = %.4f)", best_t, best_f1)
    return best_t


def find_optimal_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find the threshold that maximises Youden's J statistic (TPR − FPR).

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold (float in [0, 1]).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_t = float(thresholds[best_idx])
    logger.info(
        "Optimal Youden threshold: %.3f (J = %.4f)", best_t, j_scores[best_idx],
    )
    return best_t


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a matplotlib figure and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    labels = ["No Churn", "Churn"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="Actual",
        title=f"Confusion Matrix (threshold={threshold:.2f})",
    )

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)

    fig.tight_layout()
    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the ROC curve with AUC annotation.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    ax.plot(fpr, tpr, label=f"Model (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve")
    ax.legend()
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the Precision–Recall curve with AP annotation.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    ax.plot(recall_vals, precision_vals, label=f"Model (AP = {ap:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a calibration (reliability) curve.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins (defaults to ``plots.calibration_bins``).
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    n_bins = n_bins or plot_config.calibration_bins
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title="Calibration Curve")
    ax.legend()
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_lift_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a cumulative Lift curve.

    Lift at percentile *p* = (fraction of positives captured in top *p*%
    of scores) / (overall positive rate × *p*).

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    positive_rate = y_sorted.mean()

    cumulative_positives = np.cumsum(y_sorted)
    percentiles = np.arange(1, n + 1) / n
    lift = (cumulative_positives / np.arange(1, n + 1)) / positive_rate

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    ax.plot(percentiles, lift, label="Model")
    ax.axhline(1.0, linestyle="--", color="gray", label="Random (lift=1)")
    ax.set(xlabel="Fraction of customers (ranked by score)",
           ylabel="Lift", title="Cumulative Lift Curve")
    ax.legend()
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_gain_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a cumulative Gains curve.

    Gain at percentile *p* = fraction of all positives captured in the
    top *p*% of scores.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    total_positives = y_sorted.sum()

    cumulative_gain = np.cumsum(y_sorted) / total_positives
    percentiles = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=plot_config.figsize_small)
    ax.plot(percentiles, cumulative_gain, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(xlabel="Fraction of customers (ranked by score)",
           ylabel="Fraction of churners captured",
           title="Cumulative Gains Curve")
    ax.legend()
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_k: int = 20,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Args:
        feature_names: Feature name list.
        importances: Importance values (e.g. mean absolute SHAP).
        top_k: Number of top features to show.
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib ``Figure``.
    """
    order = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=plot_config.figsize_default)
    ax.barh(names, values, color="#3498db")
    ax.set(xlabel="Importance", title=f"Top {top_k} Feature Importances")
    fig.tight_layout()

    if output_path:
        _save_figure(fig, Path(output_path))
    return fig


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cv_scores: list[float] | None = None,
    output_dir: str | Path | None = None,
) -> EvaluationReport:
    """Run a full evaluation suite and optionally save all plots.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision boundary for hard predictions.
        cv_scores: Optional list of cross-validation ROC-AUC scores.
        output_dir: If provided, save all evaluation plots to this
            directory.

    Returns:
        ``EvaluationReport`` with metrics, optimal thresholds, and paths
        to any saved plots.
    """
    metrics = compute_metrics(y_true, y_prob, threshold)
    report_text = classification_report(
        y_true, (y_prob >= threshold).astype(int), zero_division=0,
    )

    optimal_f1 = find_optimal_threshold_f1(y_true, y_prob)
    optimal_youden = find_optimal_threshold_youden(y_true, y_prob)

    report = EvaluationReport(
        metrics=metrics,
        classification_report_text=report_text,
        cv_roc_auc_mean=float(np.mean(cv_scores)) if cv_scores else None,
        cv_roc_auc_std=float(np.std(cv_scores, ddof=1)) if cv_scores and len(cv_scores) > 1 else None,
        optimal_threshold_f1=optimal_f1,
        optimal_threshold_youden=optimal_youden,
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        plot_funcs: dict[str, Any] = {
            "confusion_matrix.png": lambda p: plot_confusion_matrix(y_true, y_prob, threshold, p),
            "roc_curve.png": lambda p: plot_roc_curve(y_true, y_prob, p),
            "pr_curve.png": lambda p: plot_precision_recall_curve(y_true, y_prob, p),
            "calibration_curve.png": lambda p: plot_calibration_curve(y_true, y_prob, output_path=p),
            "lift_curve.png": lambda p: plot_lift_curve(y_true, y_prob, p),
            "gain_curve.png": lambda p: plot_gain_curve(y_true, y_prob, p),
        }
        for filename, plot_fn in plot_funcs.items():
            path = out / filename
            plot_fn(path)
            report.plot_paths[filename] = path
            logger.info("Saved %s", path)

    logger.info(
        "Evaluation complete — ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f",
        metrics.roc_auc, metrics.pr_auc, metrics.f1,
    )
    return report

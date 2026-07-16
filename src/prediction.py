"""Prediction helpers for the churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.constants import TARGET_COLUMN
from src.model import load_featured_frame


def load_model(model_path: str | Path) -> Any:
    """Load a saved model pipeline from disk."""

    return joblib.load(model_path)


def predict_churn_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for the positive class."""

    return model.predict_proba(features)[:, 1]


def load_and_score(csv_path: str | Path, model_path: str | Path) -> pd.DataFrame:
    """Load featured data and attach model probabilities."""

    model = load_model(model_path)
    frame = load_featured_frame(csv_path)
    features = frame.drop(columns=[TARGET_COLUMN])
    frame = frame.copy()
    frame["churn_probability"] = predict_churn_probability(model, features)
    return frame
